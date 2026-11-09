"""Standalone stdlib bootstrap: register a killable WSL session before imports."""
import argparse
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--pid-file", required=True)
    parser.add_argument("--attempt")
    args = parser.parse_args()
    if os.getpgrp() != os.getpid():
        os.setsid()
    marker = Path(args.pid_file)
    temporary = marker.with_suffix(".tmp")
    temporary.write_text(json.dumps({"token": args.token, "pgid": os.getpgrp()}), encoding="utf-8")
    temporary.replace(marker)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from vtm_corpus.corpus_worker import process_corpus
    process_corpus(args.manifest, args.engine, attempt_id=args.attempt)


if __name__ == "__main__":
    main()
