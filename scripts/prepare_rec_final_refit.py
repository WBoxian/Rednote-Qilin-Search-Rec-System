"""Prepare all-past recommendation data after model selection; never read test logs."""
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from training.rec_refit_data import prepare_refit_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-path", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    print(prepare_refit_data(**vars(args)))


if __name__ == "__main__":
    main()
