#!/usr/bin/env python3
"""mooney_manifest.py -- tiny manifest helper for setup_spark.sh.

The Mooney manifest.json lives inside the official Hugging Face model repo.
setup_spark.sh downloads it at run time and calls this helper to resolve the
release file list without hardcoding shard hashes in the shell script.

Subcommands (all output goes to stdout as TSV, one file per line):

  mooney_manifest.py list manifest.json
      -> "<path>\t<size_bytes>\t<sha256>" for every entry in .files[]

The helper fails closed: a manifest without a .files list, or entries that are
missing any of path/size/sha, exit non-zero.
"""

import json
import sys


def die(msg):
    sys.stderr.write("mooney_manifest.py: %s\n" % msg)
    sys.exit(2)


def load(path):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except OSError as exc:
        die("cannot read %s: %s" % (path, exc))
    except ValueError as exc:
        die("%s is not valid JSON: %s" % (path, exc))


def cmd_list(path):
    manifest = load(path)
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        die("%s has no non-empty 'files' list" % path)
    for entry in files:
        try:
            name = entry["path"]
            size = int(entry["size_bytes"])
            sha = str(entry["sha256"]).strip().lower()
        except (KeyError, TypeError, ValueError) as exc:
            die("bad files[] entry %r: %s" % (entry, exc))
        if not name or "/" in name or name in (".", "..") or size <= 0:
            die("unsafe or invalid path/size in files[] entry %r" % name)
        if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
            die("bad sha256 %r for %s" % (sha, name))
        sys.stdout.write("%s\t%d\t%s\n" % (name, size, sha))
    return 0


def main(argv):
    if len(argv) != 3 or argv[1] != "list":
        sys.stderr.write("usage: mooney_manifest.py list MANIFEST.json\n")
        return 2
    return cmd_list(argv[2])


if __name__ == "__main__":
    sys.exit(main(sys.argv))
