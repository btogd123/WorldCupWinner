"""
Run the full preprocessing pipeline: raw data → processed features.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from features.pipeline import preprocess_pipeline

if __name__ == "__main__":
    preprocess_pipeline()
