import collections
import random
import os
import queue
import threading
import torch

import tempfile
import atexit
import shutil

# Disk cache lives in the OS temp directory to avoid cluttering the project
_REPLAY_CACHE_DIR = tempfile.mkdtemp(prefix="linefollower_replay_cache_")

def _cleanup_cache():
    try:
        shutil.rmtree(_REPLAY_CACHE_DIR)
    except OSError:
        pass
atexit.register(_cleanup_cache)


class ReplayBuffer:
    """
    Tiered Replay Buffer: Hot RAM tier + Cold Disk tier with ASYNC writes.

    Architecture
    ------------
    Hot tier  (RAM)  — holds the most recent `ram_capacity` experiences.
                       Zero disk I/O on reads; instant random access.

    Cold tier (Disk) — when the hot tier overflows, evicted experiences are
                       queued for async write to training/replay_cache/<n>.pt
                       by a background daemon thread, so push() never blocks.

    Async writes
    ------------
    A single daemon thread (self._writer_thread) drains self._write_queue.
    torch.save() is expensive (~1–5 ms per file); doing it on the main thread
    stalls the training loop. The background thread handles all file I/O while
    training continues uninterrupted.

    Race-condition safety
    ---------------------
    A path is registered in self._disk_paths only after the write is confirmed
    by the writer thread via self._written_paths (thread-safe set with a lock).
    Sampling checks this set before loading, so half-written files are never read.

    Disk location
    -------------
    training/replay_cache/<index>.pt  — created automatically, persists between runs.
    Call clear() to wipe both tiers between phases.
    """

    def __init__(self, capacity: int, ram_fraction: float = 0.05):
        """
        Parameters
        ----------
        capacity     : total experiences across both tiers.
        ram_fraction : fraction to keep in RAM (default 15 %).
                       The rest overflows async to disk.
        """
        self.capacity = capacity
        self.ram_capacity  = max(64, int(capacity * ram_fraction))
        self.disk_capacity = capacity - self.ram_capacity

        # --- Hot tier ---
        self._ram: collections.deque = collections.deque(maxlen=self.ram_capacity)

        # --- Cold tier ---
        os.makedirs(_REPLAY_CACHE_DIR, exist_ok=True)
        self._disk_paths: list = []        # ordered ring of file paths (registered when written)
        self._disk_write_idx: int = 0      # ring-buffer write pointer

        # Thread-safe set of paths that have been fully written to disk
        self._written_lock = threading.Lock()
        self._written_paths: set = set()

        # Async writer: single daemon thread drains the queue
        self._write_queue: queue.Queue = queue.Queue(maxsize=512)
        self._writer_thread = threading.Thread(
            target=self._disk_writer_loop, daemon=True, name="ReplayDiskWriter"
        )
        self._writer_thread.start()

        # Async prefetcher: pulls random items from disk into RAM before they are needed
        self._prefetch_queue: queue.Queue = queue.Queue(maxsize=200)
        self._prefetch_thread = threading.Thread(
            target=self._disk_prefetch_loop, daemon=True, name="ReplayDiskPrefetcher"
        )
        self._prefetch_thread.start()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def push(self, experience) -> None:
        """
        Add a new experience.
        If the RAM tier is full, the oldest experience is queued for async
        disk write and the new one takes its place in the hot deque.
        Returns immediately — no blocking I/O on the calling thread.
        """
        if len(self._ram) == self.ram_capacity:
            self._queue_evict(self._ram[0])   # queue BEFORE deque evicts it
        self._ram.append(experience)

    def sample(self, batch_size: int):
        """
        Return a random batch drawn proportionally from RAM and written disk files.
        Skips disk paths that haven't been fully written yet.
        Returns None if fewer than batch_size total confirmed experiences exist.
        """
        n_ram = len(self._ram)
        with self._written_lock:
            confirmed_disk = [p for p in self._disk_paths if p in self._written_paths]
        n_disk = len(confirmed_disk)
        total  = n_ram + n_disk

        if total < batch_size:
            return None

        ram_share  = min(n_ram,  max(1, round(batch_size * n_ram  / total)))
        disk_share = batch_size - ram_share

        batch = []

        if ram_share > 0 and n_ram >= ram_share:
            batch.extend(random.sample(list(self._ram), ram_share))

        if disk_share > 0 and n_disk >= disk_share:
            prefetched = []
            # Try to grab as many prefetched items as possible without blocking
            while len(prefetched) < disk_share:
                try:
                    prefetched.append(self._prefetch_queue.get_nowait())
                except queue.Empty:
                    break
                    
            batch.extend(prefetched)
            
            # If the prefetch queue was empty or too small, fall back to sync loading
            remaining = disk_share - len(prefetched)
            if remaining > 0:
                chosen = random.sample(confirmed_disk, remaining)
                for path in chosen:
                    try:
                        batch.append(torch.load(path, weights_only=False))
                    except Exception:
                        pass   # skip corrupted files

        return batch if len(batch) >= max(1, batch_size // 2) else None

    def clear(self) -> None:
        """Wipe both tiers (called between training phases)."""
        self._ram.clear()

        # Drain the writer queue before clearing disk state
        self._write_queue.join()  # wait for all pending writes to finish

        with self._written_lock:
            self._disk_paths.clear()
            self._written_paths.clear()
        self._disk_write_idx = 0

        while not self._prefetch_queue.empty():
            try:
                self._prefetch_queue.get_nowait()
            except queue.Empty:
                break

        if os.path.isdir(_REPLAY_CACHE_DIR):
            for fname in os.listdir(_REPLAY_CACHE_DIR):
                try:
                    os.remove(os.path.join(_REPLAY_CACHE_DIR, fname))
                except OSError:
                    pass

    def __len__(self) -> int:
        with self._written_lock:
            return len(self._ram) + len(self._written_paths)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _queue_evict(self, experience) -> None:
        """
        Queue an evicted experience for async disk write.
        Registers the path immediately in _disk_paths so the ring-buffer
        slot is claimed; marks it in _written_paths only after the write completes.
        """
        idx  = self._disk_write_idx % self.disk_capacity
        path = os.path.join(_REPLAY_CACHE_DIR, f"{idx}.pt")

        # Register ring slot synchronously (so index bookkeeping stays consistent)
        if len(self._disk_paths) < self.disk_capacity:
            self._disk_paths.append(path)
        else:
            # Overwriting an old slot — remove it from confirmed set until rewritten
            with self._written_lock:
                self._written_paths.discard(path)
            self._disk_paths[idx] = path

        self._disk_write_idx += 1

        # Non-blocking queue — if full, drop silently (backpressure safety)
        try:
            self._write_queue.put_nowait((path, experience))
        except queue.Full:
            pass

    def _disk_writer_loop(self) -> None:
        """Background daemon: drains write_queue and saves experiences to disk."""
        while True:
            try:
                item = self._write_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            path, experience = item
            try:
                torch.save(experience, path)
                with self._written_lock:
                    self._written_paths.add(path)
            except Exception:
                pass  # don't crash the daemon on a bad write
            finally:
                self._write_queue.task_done()

    def _disk_prefetch_loop(self) -> None:
        """Background daemon: randomly samples confirmed disk paths and pre-loads them to RAM."""
        import time
        while True:
            with self._written_lock:
                confirmed_disk = [p for p in self._disk_paths if p in self._written_paths]
            
            if not confirmed_disk:
                time.sleep(0.1)
                continue
                
            path = random.choice(confirmed_disk)
            try:
                experience = torch.load(path, weights_only=False)
                # This will block if the queue is full, which prevents RAM explosion.
                # When sample() drains the queue, this will unblock and load more.
                self._prefetch_queue.put(experience)
            except Exception:
                pass
