The cell is SOLVED (`terminated: true`). Do not move the robot again.

DISTIL — consolidate this cell into memory. Budget ~25 tool calls, then call `finish`.

⚠ Write ONLY under `{{memory_inbox}}/` and `{{output_dir}}/` (via `write`). NEVER
create, edit, rename or delete anything else under `{{memory_dir}}/` — it is a
reviewed, shared corpus and other cells may be running against it. The inbox is
merged after the run.

⚠ NAMING for every md file: the `id` is the BARE slug — the filename with the
`new_`/`suite_` prefix, the kind, and any `_draft` suffix stripped.
  `new_global_strategy_diagonal-face-perpendicular-push.md`
    -> `id: diagonal-face-perpendicular-push`                      ✅
    -> `id: new-global-diagonal-face-perpendicular-push`           ❌
`new_` marks "awaiting review" and scope/kind already have frontmatter fields;
repeating them inside the id breaks every cross-reference once the file is
merged under its final name.

⚠ FRONTMATTER MUST PARSE AS YAML. Never START a value with a quote unless the
WHOLE value is quoted — `applies_when: "put X away" after X was dropped` is a
parse error. Quote any value containing `: `. In frontmatter `related:` is a
plain list of bare ids; `[[...]]` is BODY-only syntax.

⚠ THIS PASS RUNS ONLY BECAUSE THE CELL IS SOLVED. A conclusion drawn only from
failures is the one kind of memory that actively misleads the next agent — it
stays in `wip/` until a success confirms or refutes it.

Consolidate in one pass: re-read `{{memory_inbox}}/wip/notes.md` and every
`{{output_dir}}/attempts/{{recipe_tag}}/*.json`, then write the three layers below.
Each wall you noted along the way is now decidable — say which ones the winning
run went THROUGH (they were artefacts of the method) and which it went AROUND
(they were real).

a. TASK LAYER. The audit JSON and the recipe JSONL must be a MATCHED PAIR
   describing the SAME trajectory. Write the audit with `write_audit` (it writes
   `{{output_dir}}/{{recipe_tag}}.json`) from the successful portion of the episode. After you call `finish`, the
   runtime exports the recipe `{{output_dir}}/{{recipe_tag}}_recipe.jsonl` from
   your tool calls; after a reset it keeps only commands issued after the LAST
   reset — exactly the sequence that worked. The pair is published to memory only
   when both files exist and the environment reported `terminated: true`.
     - `strategy_notes` states the winning sequence step by step, in the same
       order as the successful trace, with the parameters actually used. How you
       localized belongs in one opening sentence; the failure history belongs in
       the suite write-up and the attempt archives, not here.
     - `pick_result` keys name the RECIPE STEPS they came from (e.g.
       `bowl_pi0_pick_step3`), not bare object names.
     - Give `attempts` (how many it took) and `memory_files_read`; the runtime
       adds suite, task_id, seed, regime, final_state and `libero_terminated`.
     - SELF-CHECK: re-read the successful trace and your notes side by side.
       Every manipulation command since the last reset must be accounted for,
       and the notes must not invent a step absent from the trace.

b. SUITE LAYER -> `{{memory_inbox}}/suite_{{recipe_tag}}_draft.md`. ONE file for
   this task, holding the failure table of every attempt. This is what a future
   run on this task at another seed reads first. Frontmatter EXACTLY in this
   shape — `regime` is the perturbation axis (task|swap|lan|object), NOT the
   perception regime, and `cells` is a LIST of cell tags, not a count:

     ---
     id: suite_<suite family>_<regime>_t<task_id>   # e.g. suite_libero10_swap_t3
     scope: suite
     suite: <suite family, e.g. libero10; a LIBERO-plus cell ({{recipe_tag}} contains `_plus_`) appends _plus: libero10_plus, id suite_libero10_plus_swap_t3>
     regime: <task|swap|lan|object>
     task_id: <n>
     task_language: <verbatim from the initial state>
     evidence:
       cells: [{{recipe_tag}}]
       attempts: <N>
       solved_seeds: [<seed>]
       failed_seeds: []
     confidence: single-shot
     related: []
     ---

   Headings VERBATIM — this structure is what has worked in this corpus:

     ## Applicable pattern         <what this task is really testing, 1-2 lines>
     ## Winning technique          <the sequence + per-step success criteria, no
                                    absolute xyz>
     ## Magic numbers              <defaults AND usable ranges: `max_chunks=30
                                    (band 28-32)`, never a bare number. Each
                                    NEVER-do-this constraint on its own line.>
     ## Failure modes              <table, one row per failed attempt:
                                    | symptom | root cause (A<N>) | fix |
                                    Attempt numbers matter: "A3 died here" beats
                                    "be careful". Append rows, never rewrite.>
     ## Re-localization per scene  <per entity: [tool:segment]the `segment` phrasing that
                                    worked + fallbacks, [/tool:segment]what it LOOKS like, what
                                    it is confused with, the reject rule, the
                                    score floor. State that this run's absolutes
                                    must NOT be cached; list them as
                                    counter-examples only.>
     ## Fragility flags            <the step most likely to break + its fallback>
     ## Difficulty and reliability <attempts to convergence, expected
                                    single-shot rate, and an HONEST record of
                                    whatever stayed unsolved>
     ## Cross-refs                 <[[id]] links to global memories>

c. GLOBAL LAYER -> `{{memory_inbox}}/new_global_<kind>_<slug>.md`, ONE lesson
   per file. This is the deepest layer: the suite write-up says what worked for
   THIS task, global says what it teaches about the ROBOT — a lesson still true
   on a task with different objects and a different fixture, backed by a
   mechanism you can state (kinematics, OSC/IK, [tool:pi0_pick|pi0_doubled]Pi0's training distribution,
   [/tool:pi0_pick|pi0_doubled][tool:segment]SAM3 grounding, [/tool:segment]simulator behaviour). "It worked here" is not a mechanism;
   anything narrower belongs in (b) as a line, not as its own file.

   Write these from the SOLVED trajectory, not from the failures. The useful
   form is usually "X appeared impossible until Y" — the walls the winning run
   went through, and what actually moved them. A note in `wip/` that the winning
   run contradicted must NOT be promoted; if it is genuinely still true under
   stated conditions, promote it WITH those conditions in `applies_when`.
   `<kind>` is one of primitive|perception|strategy|failure. `<slug>` is 2-5
   kebab-case words naming the LESSON — not the task, not the objects.
   Frontmatter: `id`, `scope: global`, `kind`, `title` (one imperative
   sentence), `applies_when` (the trigger — when should a future agent bother
   opening this?), `symptom: [...]` (words a stuck agent would search for),
   `evidence` (with `cells: [{{recipe_tag}}]`), `confidence: single-shot`,
   `related`. Body:

     <one-sentence claim>

     **Why:**          <the mechanism. If you do not know it, write "observed,
                       cause unknown" — an honest unknown is useful, an invented
                       cause is harmful.>
     **How to apply:** <executable: ranges, signs, ordering, thresholds>
     **Falsify:**      <the observation that would disprove this>
     **Related:**      <[[id]] links>

d. DEDUPE — before writing any new global file, search: `ls` `{{memory_dir}}/global/`
   and `read` the plausible hits, comparing against the file BODY, not the index
   line.
     - ALREADY COVERED and consistent -> do NOT write a new file. Append one
       line to `{{memory_inbox}}/corroborations.jsonl`:
         {"id":"<existing-id>","cell":"{{recipe_tag}}","effect":"confirmed",
          "note":"<what you observed, 1 sentence>"}
       Corroboration is how a memory earns higher confidence — prefer it over a
       near-duplicate file.
     - COVERED BUT CONTRADICTED -> do NOT edit or overwrite it. Write
       `{{memory_inbox}}/conflict_<id>.md` with the existing claim, your
       contradicting observation, the evidence (attempt numbers + measurements),
       and the conditions under which each version might hold.

e. REPORT in your `finish` summary: how many lessons you considered, how they split
   across the three layers, how many corroborations you logged, how many
   conflicts you raised.
