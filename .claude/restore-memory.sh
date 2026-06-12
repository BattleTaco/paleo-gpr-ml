#!/usr/bin/env bash
# Restore the vendored Claude memory snapshot into this machine's per-project memory dir.
# Run from anywhere; it resolves the repo root from its own location.
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
slug="$(printf '%s' "$repo" | sed 's#/#-#g')"   # /Users/x/repo -> -Users-x-repo
dest="$HOME/.claude/projects/${slug}/memory"

mkdir -p "$dest"
cp "$repo/.claude/memory/"*.md "$dest/"
echo "Restored $(ls "$repo/.claude/memory/"*.md | wc -l | tr -d ' ') memory files to:"
echo "  $dest"
echo "New Claude sessions in this repo will now recall them."
