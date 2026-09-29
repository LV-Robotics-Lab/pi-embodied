The cell is SOLVED (`terminated: true`). Do not move the robot again. Run the DISTIL pass now, then call `finish`: `finish` is refused until the suite draft below exists.

Write ONLY under `{{memory_inbox}}/` and `{{output_dir}}/`. Never create, edit, rename or delete anything else under `{{memory_dir}}/`; the inbox is merged into the corpus after the run, and the next evaluation run reads the merged corpus.

Frontmatter must parse as YAML: quote any value containing `: `, and `related:` is a plain list of bare ids.

First re-read `{{memory_inbox}}/wip/notes.md` and every `{{output_dir}}/attempts/{{recipe_tag}}/*.json`: say which noted walls the winning run went THROUGH (artefacts of the method) and which it went AROUND (real).

a. TASK LAYER. Write `{{output_dir}}/{{recipe_tag}}.json`, the audit of the winning episode: {{audit_fields}}, attempts it took, final_state, the memory files read, and `strategy_notes`: the winning sequence step by step in trace order with the parameters actually used, and one opening sentence on how you localized. After `finish` the recipe `{{output_dir}}/{{recipe_tag}}_recipe.jsonl` is exported from your motion calls after the LAST reset; the audit must describe that same trajectory, every motion accounted for and none invented.

b. SUITE LAYER: `{{memory_inbox}}/suite_{{recipe_tag}}_draft.md`, one file for this task, with this frontmatter:

   ---
   id: suite_{{suite}}_{{task}}
   scope: suite
   suite: {{suite}}
   regime: default
   task_id: {{task}}
   task_language: <verbatim task text>
   evidence:
     cells: [{{recipe_tag}}]
     attempts: <N>
     solved_seeds: [{{seed}}]
     failed_seeds: []
   confidence: single-shot
   related: []
   ---

   Headings verbatim:
   ## Applicable pattern         what this task really tests, 1-2 lines
   ## Winning technique          the sequence and per-step success criteria, no absolute xyz
   ## Magic numbers              defaults and usable ranges, each never-do-this on its own line
   ## Failure modes              table `| symptom | root cause (A<N>) | fix |`, one row per failed attempt
   ## Re-localization per scene  per object: what localized it, what it is confused with, the reject rule; this run's absolutes are counter-examples only, never cached
   ## Fragility flags            the step most likely to break and its fallback
   ## Cross-refs                 [[id]] links to global memories

c. GLOBAL LAYER: `{{memory_inbox}}/new_global_<kind>_<slug>.md`, ONE lesson per file, still true on another task and backed by a mechanism you can state (kinematics, the controller, the camera, the simulator). Anything narrower is a line in (b). `<kind>` is primitive|perception|strategy|failure. Frontmatter: `id` (the bare slug), `scope: global`, `kind`, `title` (one imperative sentence), `applies_when`, `evidence` (with `cells: [{{recipe_tag}}]`), `confidence: single-shot`, `related`. Body: the claim, **Why:**, **How to apply:**, **Falsify:**.

d. DEDUPE before every global file: `ls` `{{memory_dir}}/global/` and read the plausible hits. Already covered and consistent: write no file, append `{"id":"<existing-id>","cell":"{{recipe_tag}}","effect":"confirmed"}` to `{{memory_inbox}}/corroborations.jsonl`. Contradicted: write `{{memory_inbox}}/conflict_<id>.md` instead of editing it.

e. In your `finish` summary report how many lessons you considered and how they split across the three layers.
