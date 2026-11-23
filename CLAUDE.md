# Multi-agent workflow

The main session is the **conductor**. Subagents cannot call other subagents, so the conductor starts each stage itself. Agents live in `.claude/agents/`.

| # | Agent | File | Model | Role |
|---|-------|------|-------|------|
| 1 | code-surveyor | `code-surveyor.md` | claude-haiku-5-5 | Only agent that sweeps the whole codebase |
| 2 | spec-reader | `spec-reader.md` | claude-haiku-5-5 | Only agent that reads documents (.md, .pdf, .ppt, .docx, prompts, requirements) |
| 3 | planner | `planner.md` | claude-opus-5-5 | Writes the plan for user approval |
| 4 | executor | `executor.md` | claude-sonnet-5-5 | Runs the approved plan (the working model) |
| 5 | prompt-packer | `prompt-packer.md` | claude-haiku-5-5 | Merges 1, 2, 6 into one prompt for 3 |
| 6 | research | `research.md` | claude-haiku-5-5 | Web research when needed |

## Flow

1. **Stage A (parallel):** start only the agents the task needs.
   - Code question, bug, or feature lookup → **code-surveyor**
   - User says they gave a document, .md, PDF, slides, or requirements → **spec-reader** (no other agent reads these)
   - Outside facts needed → **research**
2. **Stage B:** send all Stage A outputs, plus the user's request verbatim, to **prompt-packer**.
3. **Stage C:** send the packed brief to **planner**. Show its plan to the user.
4. **Gate:** stop. Wait for the user to approve the plan. Do not execute without approval.
5. **Stage D:** after approval, hand the approved plan to **executor** (Sonnet).

## Hard rules

- Only code-surveyor reads the whole codebase. Other agents read only the files they are given or need.
- Only spec-reader reads documents. If the user mentions a document, route it there.
- Planner never edits files. Executor never starts before approval.
- Agents pass text between them through the conductor, not directly.
- Skip any stage the task does not need, and say so in one line.
