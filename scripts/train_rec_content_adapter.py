"""Portable fixed-recipe fit-only content training. All inputs require SHA256.

Config: {"version":1,"training_cutoff_exclusive":1732696816,"inputs":
 {"requests":{"path":"...","sha256":"..."},...}}. Required asset names:
requests, labels, partitions, split_manifest, text_vectors, text_map, pca_mean,
pca_components, pca_vectors, item_ids. Paths are relative to the config file.

python scripts/train_rec_content_adapter.py prepare --config fit.json --output run
python scripts/train_rec_content_adapter.py train --output run --device cuda
python scripts/train_rec_content_adapter.py export --output run --epoch 3 --destination heads

No validation/test labels or automated model selection. Original embedding format
and source fingerprints are external contracts; cleaning text does not rewrite
existing frozen vectors. A new run does not overwrite prior results.
"""
from pathlib import Path
import argparse
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from training.rec_content_adapter import prepare, train, export

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="phase", required=True)
    p = commands.add_parser("prepare"); p.add_argument("--config", required=True, type=Path); p.add_argument("--output", required=True, type=Path)
    p = commands.add_parser("train"); p.add_argument("--output", required=True, type=Path); p.add_argument("--device", required=True, choices=["cpu", "cuda"])
    p = commands.add_parser("export"); p.add_argument("--output", required=True, type=Path); p.add_argument("--epoch", required=True, type=int, choices=[1, 2, 3]); p.add_argument("--destination", required=True, type=Path)
    args = parser.parse_args(); values = vars(args); phase = values.pop("phase")
    {"prepare": prepare, "train": train, "export": export}[phase](**values)

if __name__ == "__main__":
    main()
