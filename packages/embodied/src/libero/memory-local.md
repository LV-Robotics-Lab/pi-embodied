# Memory (local exploration corpus)
Cell `{{recipe_tag}}`. Use the local exploration corpus at `{{memory_dir}}/`. Read it with `read`, `ls`, `grep` and `find`; it cannot be written, and `_internal/` and other robots' memory cannot be read (never read `_internal/` during evaluation). Write only the audit, with `write`, to `{{output_dir}}/`. Do not re-read a file you already read in this session.

Its three layers have different jobs; use every one that is available:
- GLOBAL: `{{memory_dir}}/global/`, reusable robot, perception and primitive lessons, indexed by `{{memory_dir}}/MEMORY.md`.
- SUITE: the leaf for this task and regime under `{{memory_dir}}/suite/` (named like `suite_<suite family>_<regime>_t{{task}}.md`, e.g. `suite_libero10_swap_t{{task}}.md`): the task strategy, validated ranges and failure table.
- TASK: `{{memory_dir}}/task_only/{{reference_tag}}.json` plus `{{memory_dir}}/task_only/{{reference_tag}}_recipe.jsonl`: the matched successful audit and command order from seed 0.

1. Read each available layer first, before your first motion, in this order: the task audit, the task recipe, the matching suite leaf, then `MEMORY.md` and only the relevant global leaves. If a layer is absent, say so and continue with the others. Recipes are technique references, not coordinates: treat every absolute coordinate as stale and re-localize every entity in the current image.
2. Before `finish`, write the audit `{{output_dir}}/{{recipe_tag}}.json`: suite, task_id, seed, `regime: "strict_perception"`, `strategy_notes` (how you localized, what you did, and the exact memory files you used), `pick_result`, `final_state` (the latest `state`) and `terminated`. If the task is not solved, set `terminated: false` and say what you tried in this episode and where it stalled.
