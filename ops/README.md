# ops/ — durable reference

Facts, contracts, and protocols that change **rarely**. Living state (status, open
threads, the plan) lives in `.notes/` — not here. Evergreen rules live in `CLAUDE.md`.

| File | What it is | When to read |
|---|---|---|
| `source-of-truth.md` | Verified facts about TRIBE v2 internals | Before writing any model code |
| `interface-contracts.md` | Every function signature + data shape (keep in sync with code) | Before/during implementation |
| `claims-protocol.md` | How to curate a NeuroCheck claim | When adding/editing a claim |
| `compute-playbook.md` | Pre-scripted free-GPU session plans | Before any GPU session |
| `knowledge-gaps.md` | Open unknowns to resolve | When stuck or planning |
| `decision-log.md` | Every decision + reasoning (newest above the marker) | After any decision |
| `principal-review-2026-07-23.{pdf,html}` | The strategic review that set the current direction | For the "why" behind Corticall |
| `archive/` | Superseded process files (old war-room, progress, assessment, discussions, pre-mortem, red-team, session-protocol, three-musketeers) | History only |

## Where state went

Status, priorities, open threads, and the step-by-step plan used to live in
`war-room.md` + a heavier process ritual here. They now live in `.notes/`
(`BRIEF_ME.md` / `LOOSE-ENDS.md` / `journal/` / `plans/corticall/ROADMAP.md`).
Start every session from `.notes/BRIEF_ME.md`.
