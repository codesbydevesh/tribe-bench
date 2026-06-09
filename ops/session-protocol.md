# Session Protocol — How to Start and End Claude Sessions

Context dies when a session ends. This protocol preserves it.

---

## Opening a New Session

### Step 1: Orient (30 seconds)

Paste this to Claude at the start of every session:

```
Read ops/war-room.md. Tell me:
1. What was done last session
2. What's next in priority order
3. Any blockers
```

Claude reads CLAUDE.md automatically. The war room gives it the rest.

### Step 2: Set the Task

Either:
- **You know what to work on:** Tell Claude directly. "Today we're writing atlas.py."
- **You don't know:** Let the war room decide. It has the priority list.

### Step 3: Load Context

If the task involves:
- **Model code** → Tell Claude: "Read ops/source-of-truth.md and ops/interface-contracts.md first."
- **Claims curation** → Tell Claude: "Read ops/claims-protocol.md first."
- **GPU session prep** → Tell Claude: "Read ops/compute-playbook.md first."
- **Review/QA** → Tell Claude: "Read ops/red-team.md first."

### Step 4: Work

Do the work. Make decisions. Write code. Curate claims.

---

## Closing a Session

### Step 1: Update War Room

Tell Claude:

```
Update ops/war-room.md with what we did today. Mark completed items,
update status, add any new blockers discovered.
```

### Step 2: Log Decisions

If any decisions were made during the session:

```
Add entries to ops/decision-log.md for: [list the decisions]
```

### Step 3: Log New Knowledge Gaps

If we discovered things we don't know:

```
Add entries to ops/knowledge-gaps.md for: [list the gaps]
```

### Step 4: Write Session Summary

Ask Claude to write a summary block at the end of the session.
This is for YOUR reference between sessions.

```
Write a session summary covering:
- What was accomplished
- Key decisions made (reference decision-log.md IDs)
- Files created or modified
- What to do next session
- Any unresolved questions
```

Save this somewhere you'll see it before the next session.
(A text file, a note on your phone, whatever works.)

---

## Mid-Session Context Recovery

If Claude seems confused or is contradicting earlier decisions:

```
Read ops/decision-log.md. Decision DXXX covers this.
Follow what was decided there.
```

If Claude's context is getting long and it's making mistakes:

```
/compact — Preserve: current task, all file paths modified this session,
key decisions from decision-log.md entries DXXX-DYYY.
```

---

## Session Types

### Deep Work Session (2+ hours)
- Focus on one module or one batch of claims
- End with updated war room + decision log

### Quick Fix Session (< 30 min)
- Bug fix, small update, quick question
- Skip the full protocol. Just state the task and go.

### GPU Session (Kaggle/Colab/etc.)
- NOT a Claude session. This is you running pre-written scripts.
- Before: read compute-playbook.md for the exact script
- After: start a Claude session to process results and update source-of-truth.md

### Review Session
- Dedicated to running red-team.md checklists
- No new code or claims in this session — only reviewing existing work
- These should happen after every major milestone (10 claims, 1 module complete, etc.)

---

## What Gets Lost Between Sessions (And How to Prevent It)

| What Dies | Where It's Preserved |
|-----------|---------------------|
| "What are we working on?" | war-room.md |
| "Why did we choose X over Y?" | decision-log.md |
| "What does function Z expect?" | interface-contracts.md |
| "Is fact F verified?" | source-of-truth.md |
| "What don't we know?" | knowledge-gaps.md |
| "What could go wrong?" | pre-mortem.md |

Everything important lives in a file, not in chat.
