# MemPalace Hooks — Auto-Save for Terminal AI Tools

These hook scripts make MemPalace save automatically. No manual "save" commands needed.

## What They Do

| Hook | When It Fires | What Happens |
|------|--------------|-------------|
| **Save Hook** | Every 15 human messages | Auto-mines transcript (tool output included), then blocks the AI to save topics/decisions/quotes |
| **PreCompact Hook** | Right before context compaction | Auto-mines transcript, then emergency save — forces the AI to save EVERYTHING before losing context |

**Two-layer capture:** Hooks auto-mine the JSONL transcript directly into the palace (capturing raw tool output — Bash results, search findings, build errors). They also block the AI with a reason message telling it to save verbatim tool output and key context. Belt and suspenders — tool output gets stored even if the AI summarizes instead of quoting.

## Install — Claude Code

Add to `.claude/settings.local.json`:

```json
{
  "hooks": {
    "Stop": [{
      "matcher": "*",
      "hooks": [{
        "type": "command",
        "command": "/absolute/path/to/hooks/mempal_save_hook.sh",
        "timeout": 30
      }]
    }],
    "PreCompact": [{
      "hooks": [{
        "type": "command",
        "command": "/absolute/path/to/hooks/mempal_precompact_hook.sh",
        "timeout": 30
      }]
    }]
  }
}
```

Make them executable:
```bash
chmod +x hooks/mempal_save_hook.sh hooks/mempal_precompact_hook.sh
```

## Install — Codex CLI (OpenAI)

Add to `.codex/hooks.json`:

```json
{
  "Stop": [{
    "type": "command",
    "command": "/absolute/path/to/hooks/mempal_save_hook.sh",
    "timeout": 30
  }],
  "PreCompact": [{
    "type": "command",
    "command": "/absolute/path/to/hooks/mempal_precompact_hook.sh",
    "timeout": 30
  }]
}
```

## Configuration

Edit `mempal_save_hook.sh` to change:

- **`SAVE_INTERVAL=15`** — How many human messages between saves. Lower = more frequent saves, higher = less interruption.
- **`STATE_DIR`** — Where hook state is stored (defaults to `~/.mempalace/hook_state/`)
- **`MEMPAL_DIR`** — Optional **project directory** (code, notes, docs) to also mine on each save trigger, with `--mode projects`. The hook ALWAYS mines the active conversation transcript automatically with `--mode convos` — `MEMPAL_DIR` is purely additive, never an override. Leave blank if you don't want to ingest project files.
- **`MEMPALACE_PYTHON`** — Optional env var. Python interpreter with mempalace + chromadb installed. Auto-detects: `MEMPALACE_PYTHON` env var → repo `venv/bin/python3` → system `python3`. Set this if your venv is in a non-standard location.

### mempalace CLI

The relevant commands are:

```bash
mempalace mine <dir>               # Mine all files in a directory
mempalace mine <dir> --mode convos # Mine conversation transcripts only
```

The hooks resolve the repo root automatically from their own path, so they work regardless of where you install the repo.

## LLM-Assisted Hooks (Local Ollama)

The Stop and PreCompact hooks can call a **local** LLM (Ollama, default model `gemma4:e4b`) to enrich the metadata they save. **No content ever leaves your machine** — local-first is non-negotiable.

What the LLM does:

| Piece | Hook | What it adds |
|-------|------|--------------|
| **Themes** | Stop | LLM-extracted topic phrases joined with keyword themes in the AAAK envelope |
| **Diary suffix** | Stop | Appends `summary:.../decisions:.../blockers:.../rating:★★★★` to the CHECKPOINT line |
| **KG triples** | PreCompact | Extracts up to 12 `subject → predicate → object` triples (allow-listed predicates only, confidence ≥ 0.7) and writes them to the knowledge graph |

Verbatim drawers are **never** touched by the LLM — only metadata fields. If Ollama isn't running, every piece falls back to the deterministic path silently and the hook still completes.

### Defaults

All three pieces are **on by default** because the implementation caches a `__unavailable__` sentinel after the first failed probe — missing Ollama costs one timeout per session, not per hook fire.

### Prerequisite

```bash
ollama pull gemma4:e4b
```

Any other Ollama model works too — set `hook_llm_model` to override.

### Config keys (`~/.mempalace/config.json`)

```json
{
  "hooks": {
    "llm_themes": true,
    "llm_diary": true,
    "llm_precompact_kg": true,
    "llm_provider": "ollama",
    "llm_model": "gemma4:e4b",
    "llm_timeout_s": 10
  }
}
```

### Env overrides

Env vars take precedence over the config file:

| Variable | Effect |
|----------|--------|
| `MEMPALACE_HOOK_LLM_THEMES` | `0`/`false` to disable LLM topic extraction in Stop |
| `MEMPALACE_HOOK_LLM_DIARY` | `0`/`false` to disable the CHECKPOINT suffix in Stop |
| `MEMPALACE_HOOK_LLM_PRECOMPACT_KG` | `0`/`false` to disable KG triple extraction in PreCompact |
| `MEMPALACE_HOOK_LLM_PROVIDER` | Provider name (currently only `ollama`) |
| `MEMPALACE_HOOK_LLM_MODEL` | Model identifier passed to the provider |
| `MEMPALACE_HOOK_LLM_TIMEOUT_S` | Per-call timeout in seconds (integer) |

Quick disable for one session:

```bash
MEMPALACE_HOOK_LLM_THEMES=0 \
MEMPALACE_HOOK_LLM_DIARY=0 \
MEMPALACE_HOOK_LLM_PRECOMPACT_KG=0 \
claude
```

### KG predicate allow-list

Only these predicates are accepted from the LLM for PreCompact KG extraction (anything else is dropped):

`works_on`, `assigned_to`, `blocked_by`, `decided`, `decides`, `mentions`, `discusses`, `depends_on`, `related_to`, `owns`, `uses`, `uses_tool`, `reports_to`, `asked_about`, `completed`

Subject and object strings are run through `sanitize_kg_value`; predicate through `sanitize_name`. Triples are written with `valid_from = today` and `source_closet = "precompact-hook"`.

## How It Works (Technical)

### Save Hook (Stop event)

```
User sends message → AI responds → Claude Code fires Stop hook
                                            ↓
                                    Hook counts human messages in JSONL transcript
                                            ↓
                              ┌─── < 15 since last save ──→ echo "{}" (let AI stop)
                              │
                              └─── ≥ 15 since last save
                                            ↓
                                    Auto-mine transcript → palace (tool output captured)
                                            ↓
                                    {"decision": "block", "reason": "save tool output verbatim..."}
                                            ↓
                                    AI saves to palace (topics, decisions, quotes)
                                            ↓
                                    AI tries to stop again
                                            ↓
                                    stop_hook_active = true
                                            ↓
                                    Hook sees flag → echo "{}" (let it through)
```

The `stop_hook_active` flag prevents infinite loops: block once → AI saves → tries to stop → flag is true → we let it through.

### PreCompact Hook

```
Context window getting full → Claude Code fires PreCompact
                                        ↓
                                Find transcript (from input or session_id lookup)
                                        ↓
                                Auto-mine transcript → palace (tool output captured)
                                        ↓
                                {"decision": "block", "reason": "save tool output verbatim..."}
                                        ↓
                                AI saves everything
                                        ↓
                                Compaction proceeds
```

No counting needed — compaction always warrants a save. The auto-mine captures raw tool output before the AI gets a chance to summarize it away.

## Debugging

Check the hook log:
```bash
cat ~/.mempalace/hook_state/hook.log
```

Example output:
```
[14:30:15] Session abc123: 12 exchanges, 12 since last save
[14:35:22] Session abc123: 15 exchanges, 15 since last save
[14:35:22] TRIGGERING SAVE at exchange 15
[14:40:01] Session abc123: 18 exchanges, 3 since last save
```

## Known Limitations

**Hooks require session restart after install.** Claude Code loads hooks from `settings.json` at session start only. If you run `mempalace init` or manually edit hook config mid-session, the hooks won't fire until you restart Claude Code. This is a Claude Code limitation.

**`MEMPAL_PYTHON` override for the hook's internal Python calls.** The save hook parses its JSON input and counts transcript messages with `python3`. When the harness is launched from a GUI on macOS — `open -a`, Spotlight, the dock — its `PATH` is the minimal `/usr/bin:/bin:/usr/sbin:/sbin` inherited from `launchd`, not your shell PATH. If `python3` isn't on that PATH, those internal calls fail and the hook can't count exchanges.

Point the hook at any Python 3 interpreter to fix it:

```bash
export MEMPAL_PYTHON="/usr/bin/python3"                   # system Python is fine
export MEMPAL_PYTHON="$HOME/.venvs/mempalace/bin/python"  # or your venv
```

Resolution priority: `$MEMPAL_PYTHON` (if set and executable) → `$(command -v python3)` → bare `python3`. The interpreter only needs `json` and `sys` from the standard library — `mempalace` itself does not need to be installed in it.

Note: the `mempalace mine` auto-ingest runs via the `mempalace` CLI, so that command also needs to be on the hook's `PATH`. Installing with `pipx install mempalace` or `uv tool install mempalace` puts it on a stable global location; otherwise extend the hook environment's `PATH` to include your venv's `bin/`.

## Backfill Past Conversations

The hooks only capture conversations going forward. To mine **past** Claude Code sessions into your palace, run a one-time backfill:

```bash
mempalace mine ~/.claude/projects/ --mode convos
```

This scans all JSONL transcripts from previous sessions and files them into the `conversations` wing. On a typical developer machine with months of history, this can yield 50K–200K drawers.

For Codex CLI sessions:
```bash
mempalace mine ~/.codex/sessions/ --mode convos
```

This only needs to be done once — after that, the hooks auto-mine each session as you go.

## Cost

**Zero extra tokens.** The hooks notify the AI that saves happened in the background — the AI doesn't need to write anything in the chat. All filing is handled automatically. Previous versions asked the AI to write diary entries and drawer content in the chat window, which cost ~$1/session in retransmitted tokens.
