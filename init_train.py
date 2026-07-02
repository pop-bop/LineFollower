import os
import sys

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

from training.train import run_training

if __name__ == '__main__':
    run_training()