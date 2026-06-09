# ops/ — The Operating System

This directory contains 10 operational files that govern how we work on tribe-bench.
They are not documentation. They are protocols, contracts, and living records.

| # | File | What It Does | When to Use |
|---|------|-------------|-------------|
| 1 | `war-room.md` | Status of everything, what's next | START of every session |
| 2 | `source-of-truth.md` | Verified facts about TRIBE v2 internals | Before writing any model code |
| 3 | `interface-contracts.md` | Every function signature and data shape | Before and during implementation |
| 4 | `claims-protocol.md` | How to curate NeuroCheck claims | When adding any claim |
| 5 | `compute-playbook.md` | Pre-scripted GPU session plans | Before any GPU session |
| 6 | `red-team.md` | Adversarial review checklists | Before shipping anything |
| 7 | `decision-log.md` | Record of every decision with reasoning | After any decision |
| 8 | `session-protocol.md` | How to start and end Claude sessions | Every session open/close |
| 9 | `knowledge-gaps.md` | What we don't know and need to learn | When stuck or planning |
| 10 | `pre-mortem.md` | Risk analysis: what kills each build | Before starting a build |

## The Workflow

```
SESSION START
  │
  ├─ Claude reads CLAUDE.md (automatic)
  ├─ Read war-room.md (first action)
  ├─ Human says what to work on (or war-room decides)
  │
  ├─ BEFORE CODING:
  │   ├─ Check source-of-truth.md (are our assumptions verified?)
  │   ├─ Check interface-contracts.md (what's the function spec?)
  │   └─ Check pre-mortem.md (what could go wrong?)
  │
  ├─ DURING CODING:
  │   ├─ Follow claims-protocol.md (if curating claims)
  │   ├─ Follow compute-playbook.md (if on GPU)
  │   └─ Log decisions in decision-log.md
  │
  ├─ BEFORE SHIPPING:
  │   └─ Run red-team.md checklist
  │
  SESSION END
  │
  ├─ Update war-room.md (status changes)
  ├─ Update decision-log.md (new decisions)
  ├─ Update knowledge-gaps.md (new gaps found)
  └─ Follow session-protocol.md closing checklist
```
