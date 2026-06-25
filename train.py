#!/usr/bin/env python3
"""
CHAP train entry point — no-op for the pre-trained heat-mortality model.

The ClimSocAna checkpoint is committed to the repo at model_artifacts/.
This script simply writes a JSON config pointing to those paths so that
predict.py can locate them at inference time.
"""
import json, sys, os

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def train(train_data_path: str, model_path: str) -> None:
    config = {
        "checkpoint": os.path.join(REPO_ROOT, "model_artifacts", "trained_state.ckpt"),
        "mort_dir":   os.path.join(REPO_ROOT, "model_artifacts", "death_cases"),
    }
    with open(model_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Model config written to {model_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python train.py <train_data> <model_path>")
        sys.exit(1)
    train(sys.argv[1], sys.argv[2])
