"""Build an immutable metadata lookup index from static notes only."""
import argparse
import json
from pathlib import Path
from backend.online.note_metadata import build_note_metadata_index

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--notes-dir', type=Path, default=Path('datasets/notes'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_note_metadata_index(args.notes_dir, args.output)), flush=True)
