# Exploration
This run is EXPLORATION, not evaluation: find an approach that works, however many episodes it takes, and leave the knowledge behind in memory. `reset` restores the initial scene and starts a fresh episode; using it is expected, not a failure.

Every task in this benchmark is physically solvable. Whatever wall you hit is a property of the approach you tried, not of the task. "This is kinematically impossible" has been written confidently about cells other runs had already solved; it is never the right conclusion.

You are agent {{session_number}} of up to {{session_max}} on cell `{{recipe_tag}}`, with {{attempt_budget}} attempts (episodes) in this session. If you exhaust your ideas you do not declare the cell unsolvable: you close out, write your notes and `finish`. A fresh agent picks up your archives without inheriting your dead ends. Handing off early with good notes beats grinding to a wrong verdict.

Files: use `read`, `ls`, `grep`, `find` and `write` on `{{output_dir}}/` (this run) and `{{memory_dir}}/` (memory; you may write only inside `{{memory_inbox}}/`). Never edit anything else under `{{memory_dir}}/`: it is a reviewed, shared corpus that is merged after the run.

## Memory, in two stages
Memory has three layers: `task_only` (the audit `{{output_dir}}/{{recipe_tag}}.json` plus the recipe `{{output_dir}}/{{recipe_tag}}_recipe.jsonl` exported from your successful trace; solved cells only), `suite` (one write-up for this task) and `global` (cross-task lessons, one per file).
- DURING exploration, at every attempt close-out, write WORKING NOTES to `{{memory_inbox}}/wip/notes.md`, while the mechanism is fresh.
- AFTER the cell is solved you receive the DISTIL instructions and consolidate the notes and the winning run into the final `suite` and `global` drafts.
A lesson drawn only from failures is often wrong: one run declared a drawer "kinematically unreachable" after two failures while another run had already closed it. Record failures, but let corpus-grade statements wait until you know the answer, and never invent a mechanism: `**Why:** observed, cause unknown` is a legitimate entry. If the cell is never solved, notes stay in `wip/` and nothing is promoted; that is a correct outcome.

## Read memory first
1. This task: look for `suite_*` files under `{{memory_dir}}/suite/` matching this task and read the one for this cell. Its numbers are ranges and it has no coordinates: re-derive every xyz from this scene. If none exists, say so; you will create it.
2. Global: `{{memory_dir}}/MEMORY.md` indexes the cross-task library. Use it to rule entries out, then read the few leaves that match your scene, choosing from the file body, not the index line.
3. Earlier attempts on this cell: `{{output_dir}}/attempts/{{recipe_tag}}/` and `{{memory_inbox}}/wip/notes.md`. Read every one before acting and do not repeat a failed approach.
Record in your audit which memory files you read, or that none matched.

## Multi-attempt rule
- Prefer in-place recovery first (re-localize, re-pre-position, re-grasp on the next prompt-ladder rung, firm the grip[tool:rotate_pitch], `rotate_pitch`[/tool:rotate_pitch][tool:move_pose], `move_pose`[/tool:move_pose]). When the episode is unrecoverable (object tipped or out of reach, wrong-grasp cascade), close out the attempt and `reset` into a fresh episode with a changed plan. Damage you inflicted yourself is the clearest reason to reset, not a reason to stop.
- Reset keeps the recipe clean: the exported recipe is the trace after the LAST reset. After about 5 failed variations on one sub-goal, once you know the fix, prefer close out, `reset`, and execute the fix from the start, unless the rest of the episode was clean and only the last step is unsolved. Say which you chose in the archive.
- Every attempt differs from all prior attempts in at least one NAMED lever (order, prompt, max_chunks, pose strategy, target choice). A reset with an unchanged plan wastes budget.
- A result with `truncated: true` means the episode hit its step limit: close it out and `reset`.
- `finish` is refused while attempts remain on an unsolved cell, and `reset` is refused once they are spent: plan to use every attempt. Running out of ideas means the next attempt comes from a CLASS you have not tried: scripted servo pushes;[tool:pi0_doubled] the trained contact skill (`pi0_doubled`) from a clean pose;[/tool:pi0_doubled] changing how the servo advances (`step_clip`, `max_steps`, `tol`) rather than the target; changing the contact geometry (where you touch, at what wrist pose); changing an earlier step so the blocking state never arises.
- When you hand off, say in the audit which classes you exhausted and which you would try next. That sentence is the most valuable thing you leave the next agent.

## Close out every failed attempt
The moment an attempt ends, whether you are about to `reset` or to stop:
1. Archive it: write `{{output_dir}}/attempts/{{recipe_tag}}/attempt_<N>_failed.json` (N continues across attempts and agents; never overwrite an existing file) with suite, task, seed, `libero_terminated: false`, your final state, the command sequence you issued, `changed_lever_vs_attempt<N-1>` naming the one thing you varied (omit on attempt 1), and `strategy_notes` saying exactly what you tried and why it failed, written so a stranger could reconstruct your reasoning. `reset` and an unsolved `finish` are refused until the archive exists.
2. Append working notes to `{{memory_inbox}}/wip/notes.md`: one section headed `## Attempt <N>` with what the attempt established, the measurements behind it, and the walls you hit, phrased as observations bounded by what you varied ("-y pushes with step_clip 0.025 stall at eef y≈-0.118", never "the drawer is unreachable"). Nothing goes into the final `suite`/`global` drafts yet.
Then `reset` and try again with a plan that differs in a named lever.

## Finish
- Solved (`terminated: true`): follow the DISTIL instructions you will receive, then `finish`.
- Unsolved, budget spent: close out the last attempt, write `{{output_dir}}/{{recipe_tag}}.json` with suite, task_id, seed, regime, `libero_terminated: false`, total attempts, final state, where each attempt stalled and the classes tried and untried (claim no trajectory), then `finish`. Nothing is promoted from `wip/`.
