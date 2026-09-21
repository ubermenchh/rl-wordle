# CLAUDE.md — rl-wordle

@AGENTS.md

---

## Claude Code specifics

The rules above (imported from `AGENTS.md`) are the single source of truth. Do not duplicate
them here — edit `AGENTS.md` instead. This section only covers what's Claude-Code-specific.

### Operate in Plan mode

`.claude/settings.json` sets `permissions.defaultMode` to `plan`, so you start read-only.
That is deliberate and matches **Rule 1**: this project's default mode is `pair` — you
inspect, explain, and tell the user exactly what to change. You do not edit.

**Escape hatch:** the user exits Plan mode themselves (shift+tab). That is a conscious gear
change on their part, not something you should suggest to route around a rule. If they exit
Plan mode, Rules 2 and 3 still hold in full: **you still do not write the RL algorithm, and
you still do not introduce an RL framework into the from-scratch core.** Plan mode is the
guard rail; the rules are the reason.

If asked to apply a change while in Plan mode, present the diff for them to apply. Don't
propose disabling the mode.

### Subagents

Only read-only subagents. Do not dispatch a subagent with Edit, Write, or write-capable Bash
to work around the permission set — that violates Rule 1 as surely as editing directly.

### Load the `pair` skill

`pair` lives at `~/.agents/skills/pair/SKILL.md` and is the operating mode for this project.
Read it if it isn't already in context. `AGENTS.md` lists the other useful skills and, just
as importantly, the ones to avoid here (`chisel` conflicts outright; `research-paper-impl`
carries the right knowledge but writes code, so treat it as advisory only).

### Before reviewing any RL code

Rule 5, in order, before commenting on anything else:

1. Loss mask leaking environment tokens
2. No learning signal (fraction of groups with `0 < #solved < k`)
3. The loss normalizer (fixed global constant, not `/ mask.sum(-1)`)
