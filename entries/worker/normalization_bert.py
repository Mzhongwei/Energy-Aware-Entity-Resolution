import argparse
import os
import sys
from pathlib import Path

"""
mode: incremental + bert
disabled: k8s BERT normalization worker interface is not implemented yet.
"""

def main():
    parser = argparse.ArgumentParser(description="Worker entry for BERT normalization.")
    parser.add_argument("--config", default="/app/config/examples/config-embedding.yaml")
    parser.add_argument("--mode", default="bert-inference-inc")
    args = parser.parse_args()
    pass


if __name__ == "__main__":
    main()
