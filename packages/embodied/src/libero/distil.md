The cell is SOLVED (`terminated: true`). Do not move the robot again. Run the DISTIL pass now (about 25 tool calls), then call `finish`.

Write ONLY under `{{memory_inbox}}/` and `{{output_dir}}/`. Never create, edit, rename or delete anything else under `{{memory_dir}}/`; the inbox is merged after the run.

Naming: a draft's `id` is the BARE slug, the filename without the `new_`/`suite_` prefix, the kind and any `_draft` suffix (`new_global_strategy_diagonal-face-perpendicular-push.md` has `id: diagonal-face-perpendicular-push`). Frontmatter must parse as YAML: never start a value with a quote unless the whole value is quoted, quote any value containing `: `, and `related:` is a plain list of bare ids (`[[id]]` is body-only syntax).

First re-read `{{memory_inbox}}/wip/notes.md` and every `{{output_dir}}/attempts/{{recipe_tag}}/*.json`. Each wall noted along the way is now decidable: say which ones the winning run went THROUGH (artefacts of the method) and which it went AROUND (real).

a. TASK LAYER. Write `{{output_dir}}/{{recipe_tag}}.json`, the audit of the winning episode. After `finish`, the recipe `{{output_dir}}/{{recipe_tag}}_recipe.jsonl` is exported from your tool calls after the LAST reset; the audit must describe that same trajectory.
   - `strategy_notes`: the winning sequence step by step, in trace order, with the parameters actually used; one opening sentence on how you localized. Failure history belongs in the suite write-up.
   - `pick_result` keys name the recipe steps they came from (e.g. `bowl_pi0_pick_step3`), not bare object names.
   - suite, task_id, seed, `regime: "strict_perception"`, final_state, `libero_terminated: true`, attempts it took, memory files read.
   - Self-check: every manipulation command since the last reset is accounted for, and no step is invented.

b. SUITE LAYER: `{{memory_inbox}}/suite_{{recipe_tag}}_draft.md`, one file for this task. Frontmatter exactly in this shape (`regime` is the perturbation axis task|swap|lan|object, not the perception regime; `cells` is a list):

   ---
   id: suite_<suite family>_<regime>_t<task_id>   # e.g. suite_libero10_swap_t3
   scope: suite
   suite: <suite family, e.g. libero10>
   regime: <task|swap|lan|object>
   task_id: <n>
   task_language: <verbatim task text>
   evidence:
     cells: [{{recipe_tag}}]
     attempts: <N>
     solved_seeds: [<seed>]
     failed_seeds: []
   confidence: single-shot
   related: []
   ---

   Headings verbatim:
   ## Applicable pattern         what this task really tests, 1-2 lines
   ## Winning technique          the sequence and per-step success criteria, no absolute xyz
   ## Magic numbers              defaults AND usable ranges (`max_chunks=30 (band 28-32)`), each never-do-this on its own line
   ## Failure modes              table `| symptom | root cause (A<N>) | fix |`, one row per failed attempt
   ## Re-localization per scene  per entity: the `segment` phrasing that worked and fallbacks, what it looks like, what it is confused with, the reject rule, the score floor; this run's absolutes are counter-examples only, never cached
   ## Fragility flags            the step most likely to break and its fallback
   ## Difficulty and reliability attempts to convergence, expected single-shot rate, an honest record of what stayed unsolved
   ## Cross-refs                 [[id]] links to global memories

c. GLOBAL LAYER: `{{memory_inbox}}/new_global_<kind>_<slug>.md`, ONE lesson per file, still true on a task with other objects and fixtures and backed by a mechanism you can state (kinematics, OSC/IK, Pi0's training distribution, SAM3 grounding, simulator behaviour). Anything narrower is a line in (b). Write from the solved trajectory: the useful form is usually "X appeared impossible until Y". A `wip/` note the winning run contradicted is not promoted; if it still holds under stated conditions, put them in `applies_when`. `<kind>` is primitive|perception|strategy|failure; `<slug>` is 2-5 kebab-case words naming the lesson. Frontmatter: `id`, `scope: global`, `kind`, `title` (one imperative sentence), `applies_when`, `symptom: [...]` (words a stuck agent would search for), `evidence` (with `cells: [{{recipe_tag}}]`), `confidence: single-shot`, `related`. Body:

   <one-sentence claim>

   **Why:**          the mechanism, or "observed, cause unknown"
   **How to apply:** ranges, signs, ordering, thresholds
   **Falsify:**      the observation that would disprove this
   **Related:**      [[id]] links

d. DEDUPE before every global file: `ls` `{{memory_dir}}/global/` and read the plausible hits, comparing bodies, not index lines.
   - Already covered and consistent: write no file; append one line to `{{memory_inbox}}/corroborations.jsonl`: `{"id":"<existing-id>","cell":"{{recipe_tag}}","effect":"confirmed","note":"<one sentence>"}`.
   - Covered but contradicted: do not edit it; write `{{memory_inbox}}/conflict_<id>.md` with the existing claim, your contradicting observation, the evidence (attempt numbers, measurements) and the conditions under which each version might hold.

e. In your `finish` summary report how many lessons you considered, how they split across the three layers, and how many corroborations and conflicts you logged.
