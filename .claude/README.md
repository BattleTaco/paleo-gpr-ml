# .claude (portable agent context)

This folder makes my Claude Code setup travel with the repo, so a fresh machine picks up the
same context instead of starting cold.

## What is here

- `skills/` project skills. Claude Code auto-discovers skills in a repo's `.claude/skills/`, so
  these work on any machine that has the repo, no setup needed:
  - `research-integrity-check` the pre-result checklist to run before claiming any finding.
  - `paleo-research-log` logs a finding into the journal or Obsidian in my voice.
- `memory/` a snapshot of my per-project Claude memory (the same files Claude keeps under
  `~/.claude/projects/<project-slug>/memory/`). This is a copy for transfer. See restore below.
- `settings.local.json` is machine-specific and git-ignored on purpose.

## How the context transfers

- **CLAUDE.md** (repo root) is the main portable memory. Claude reads it automatically on any
  machine. It has the goals, priorities, voice rules, repo structure, and CUDA setup.
- **Skills** in `skills/` load automatically once the repo is cloned.
- **Memory files** in `memory/` do NOT auto-load, because Claude reads memory from a per-machine
  path (`~/.claude/projects/<slug>/memory/`) that depends on where the repo sits. Restore them
  with the script below.

## Restore the memory on a new machine

From the repo root:

```bash
bash .claude/restore-memory.sh
```

It copies `memory/*.md` into this machine's `~/.claude/projects/<slug>/memory/`, where `<slug>`
is the repo's absolute path with `/` replaced by `-`. After that, new Claude sessions in this
repo recall the same memories.

## Keeping the snapshot current

When the live memory changes, re-copy it into the repo so the snapshot does not go stale:

```bash
slug="$(pwd | sed 's#/#-#g')"
cp "$HOME/.claude/projects/${slug}/memory/"*.md .claude/memory/
```
