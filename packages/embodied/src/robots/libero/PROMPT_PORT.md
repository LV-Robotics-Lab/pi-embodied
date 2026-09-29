# LIBERO prompt port (RPent → pi-embodied)

Source: RPent `eecf206` (github.com/RLinf/RPent, Apache-2.0), `robots/libero/prompts/{evaluate,explore,user}.py` and
`robots/libero/guides/*.md`. `--libero-prompt rpent` (the default) runs this port; `--libero-prompt compact` runs the
short prompt pi-embodied used before (`compact/`), kept for comparison. `result.json` records `libero_prompt`, and
`eval.sh` refuses to mix the two in one out dir (a result without the field ran `compact`).

The content is RPent's, section by section and in RPent's order; only references to things pi-embodied does not
have were rewritten. `test/libero-prompt.test.ts` checks the key rule sentences of every original section
(`test/fixtures/libero-rpent-prompt.json`) in the rendered prompt.

## How the templates render (`index.ts` `renderRpent`)

| Marker | Meaning | RPent equivalent |
| --- | --- | --- |
| `# TITLE` | one prompt section | a key of the `system_prompt()` mapping (rendered with `═══` rules) |
| `#. ` | a workflow step, numbered in order | `Numbered(WORKFLOW_STEPS)` |
| `[memory:hf]` / `[memory:local]` | kept for that `--memory-profile` | `system_prompt(variables)` choosing `LOCAL_MEMORY_PROFILE` / `LOCAL_WORKFLOW_STEPS` |
| `[part:x]` in SYSTEM.md, `[include:x]` in explore.md | a section or step both prompts share | `explore.py` importing `evaluate as base` |
| `[tool:x]` / `[tool:!x]` | kept while tool `x` is active / inactive (robot.ts `toolSections`) | none: RPent always has every tool |
| `{{memory_dir}}`, `{{reference_tag}}`, `{{recipe_tag}}`, `{{output_dir}}`, `{{memory_inbox}}` | memory.ts `render` | same variables |
| `{{suite}}`, `{{task}}`, `{{seed}}`, `{{guides_dir}}` | the cell (`index.ts` `cellVars`) | `user.py` CELL variables; guide paths were repo-relative |
| `{{session_number}}`, `{{session_max}}`, `{{attempt_budget}}` | explore.ts | same (budget is pi's `--explore-attempts-per-session`) |

With every tool active, the rendered text is RPent's. `[tool:!x]` fallbacks only appear when a tool is excluded
(`--exclude-tools`, units/code pure modes), e.g. "re-grasp a missed grasp" instead of the Pi0 prompt ladder.

## evaluate.py → SYSTEM.md

| RPent (evaluate.py) | pi-embodied (SYSTEM.md) | Replacements |
| --- | --- | --- |
| `ROLE_AND_EVALUATION` :23 | `# ROLE AND EVALUATION` | "MUST NOT call `reset`" → "MUST NOT reset or restart" (no `reset` tool outside exploration); override also covers the guides |
| `LOCAL_MEMORY_PROFILE` :533 (local profile, second section) | `[memory:local] # MEMORY PROFILE — LOCAL SUITE + TASK + GLOBAL` | adds the LIBERO-plus suite-leaf name (`suite_libero10_plus_…`, memory.ts cell tags) |
| `PROVEN_LEVERS` :40 | `# PROVEN LEVERS & LESSONS …` (`[part:proven-levers]`) | `agentview_high.png` → the `agentview_high` image; Pi0/`set_gripper`/`move_pose`/`rotate_pitch`/`pi0_doubled`/`segment` sentences in tool blocks. The local profile's shorter heading ("PROVEN LEVERS") uses the HF heading |
| `RUNTIME` :154 | `# RUNTIME` (`[part:runtime]`) | `env_server.py` + "Pi0.5 loaded" → the env server and the attached Pi0.5 server; tool list `read_text_file`/`write_text_file`/`list_dir` → `read`/`write`/`ls` (+ `grep`, `find`); "one tool at a time" is stated as a runtime guarantee; state record fields (`step`, `artifacts`, `log`) → `state_step`, `step` (env steps), `result`, `images`; the artifact key list (`*_policy.png`, `*.npz`, `*_metadata.json`) → "each state record keeps the images and, internally, depth, world map and calibration", with a note that pi shows no separate policy image |
| `GOAL` :185 | `# YOUR GOAL` | "NO `reset`" → "NO reset" |
| `RULES` :189, Rules 0–3 | `# RULES (NON-NEGOTIABLE)` (`[part:rules]`) | image names as above; Rule 1b's hint sentence has a `[tool:!pi0_pick]` fallback; Rule 2b/2c `back_project` in tool blocks |
| — | Rule 1a (inside RULES) | pi addition: the third-party grasp policies (`openvla_act`, `openvla_oft_act`, `gr00t_act`), shown only when mounted |
| `RULES` Rule 4 :274 | Rule 4 (outside the part, so explore can replace it) | "Do NOT call `reset`" → "Do NOT reset" |
| `LOCALIZATION` :287 | `# LOCALIZATION …` (`[part:localization]`) | "prefer `agentview_high.png`, fall back to `agentview.png`" → the 1024 image every state shows; `step: NN` explained as the result's `state_step`; `resolution:"low"` only for the latest state (history keeps the 1024 maps); region mode (`row_range`/`col_range`) added, from the compact prompt's rule 2; a `segment` alternative line |
| `PERCEPTION_ALGORITHM` :316 | `# FIRST-STEP ALGORITHM …` (`[part:first-step]`) | image names; step 3 adds the `segment` alternative; "median back_project" in the table → "median world xyz" |
| `WORKFLOW_STEPS[0]` read memory :386 | `[memory:hf] #. READ MEMORY FIRST` | `read_text_file`/`list_dir` → `read`/`ls`; "if a shell / grep tool is available" → pi's `grep` and `find` tools; adds "do not re-read" (compact memory-hf.md) |
| `STEP_READ_LOCAL_MEMORY` :548 | `[memory:local] #. READ EACH AVAILABLE LOCAL MEMORY LAYER FIRST` | names the file tools |
| `WORKFLOW_STEPS[1]` guides :413 | `#. READ THE GUIDES` (`[part:step-guides]`) | `robots/libero/guides/…` → `{{guides_dir}}/…` (readable through the memory guard's `readable`); one added sentence mapping RPent tool/artifact names in the guides to pi's and saying this prompt's attempt rules win |
| `WORKFLOW_STEPS[2]` seed-0 refs :419 | `[memory:hf] #. READ SEED-0 …` (`[part:step-seed0]`) | none |
| `STEP_INSPECT_INITIAL` :428 | `#. INSPECT INITIAL STATE` (`[part:step-inspect]`) | image names |
| `STEP_PERCEPTION_PASS` :433 | `#. RUN THE MANDATORY PRE-TASK PERCEPTION PASS` (`[part:step-perception]`) | none |
| `STEP_EXECUTE` :442 | `#. EXECUTE …` (`[part:step-execute]`) | "log" → `result`; adds the 0.30 m xy rule from `move_to`'s description |
| `STEP_PRIMITIVES` :452 | `#. ALLOWED PRIMITIVES` | "`reset` is FORBIDDEN" → "Resetting is FORBIDDEN"; INFRA NOTE verbatim in a `pi0_doubled` block; SAM3 `segment` aid in `[part:aids]` (it runs on the image of `step`, returns `segment_artifact`/`overlay_artifact` as in RPent) |
| — | `[part:aids]`: `preview_reach`, planned grasps, geometry (`view_points`, `mark_point`, `move_grip`) | pi additions (`--ik`, `--graspnet` …, `--geometry`), only when their tools are active; the grasp text is the compact prompt's, the geometry text summarizes OpenETA's openeta-for-codex tools |
| `STEP_RECOVERY` :489 | `#. RECOVERY` | none |
| `STEP_FINISH` :496 | `#. WHEN top-level terminated …` | "write audit" is a `write_audit` call (../../capabilities/memory): the model gives terminated, strategy_notes, memory_files_read and pick_result; the runtime adds suite, task_id, seed, regime and final_state |
| `KEY_HYPERPARAMETERS` :507 | `# KEY HYPERPARAMETERS` | none |
| `OUTPUT_DISCIPLINE` :514 | `# OUTPUT DISCIPLINE` | none |
| `user.py` CELL, MODE, BEGIN | `# CELL` and the two closing paragraphs | pi's user message is eval.sh's "Solve the task.", so the cell block lives in the system prompt; the recipe is exported by the runtime after `finish` |

## explore.py → explore.md

`--explore` with `rpent`: the robot's own prompt is empty and explore.ts appends explore.md as the whole system
prompt (RPent assembles the explore prompt separately for the same reason: explore.py's docstring). The DISTIL pass
is sent as a message once `terminated` is reported (explore.ts), not kept in the system prompt.

| RPent (explore.py) | pi-embodied | Replacements |
| --- | --- | --- |
| `ROLE` :54 | `# ROLE AND MODE` | attempt budget named (`{{attempt_budget}}`); one added sentence: the single-attempt wording of the shared PROVEN LEVERS and the guides is the evaluation's |
| `base.PROVEN_LEVERS`, `base.RUNTIME` | `[include:proven-levers]`, `[include:runtime]` | as in SYSTEM.md |
| `GOAL` :74 | `# YOUR GOAL` | `state.libero_terminated` → top-level `terminated`; "the runner exports" → "the runtime exports"; "grown at every attempt" dropped for `suite` (pi writes the suite draft in DISTIL); the file-tool paragraph from the compact explore prompt (write only in the inbox) |
| `_rules()` (RULES with `RULE_4` :116) | `[include:rules]` + explore's Rule 4 | none beyond tool blocks |
| `base.LOCALIZATION`, `base.PERCEPTION_ALGORITHM` | `[include:localization]`, `[include:first-step]` | as in SYSTEM.md |
| `STEP_READ_MEMORY` :181 | `#. READ MEMORY FIRST` | file tool names; adds (c) earlier attempts on this cell (`{{output_dir}}/attempts/{{recipe_tag}}/`, `wip/notes.md`), which pi's handoff sessions rely on (compact explore prompt) |
| `BASE_STEP_READ_GUIDES`, `…SEED0_REFS`, `…INSPECT_INITIAL`, `…PERCEPTION_PASS`, `…EXECUTE` | `[include:step-…]` | as in SYSTEM.md |
| `STEP_PRIMITIVES` :205 | `#. ALLOWED PRIMITIVES` + `[include:aids]` | the aids (segment, preview_reach, planned grasps) are included so an exploration run describes the same tools |
| `STEP_RECOVERY` :213 | `#. RECOVERY` | none |
| `STEP_CLOSE_OUT` :221 | `#. CLOSE OUT EVERY FAILED ATTEMPT` | archive path `attempts/attempt_<N>_failed.json` → `attempts/{{recipe_tag}}/attempt_<N>_failed.json` (explore.ts); "`reset` and an unsolved `finish` are refused until the archive exists" (explore.ts enforces it) |
| `STEP_FINISH` :251 | `#. WHEN top-level terminated …` | RPent's stale "Rule 4 (a)/(b)/(c) conditions" → "your attempt budget is spent (Rule 4)"; "Run the DISTIL pass below" → its instructions arrive when `terminated` is reported |
| `STEP_DISTIL` :263 | `distil.md` (sent by explore.ts) | `write_text_file`/`list_dir`/`read_text_file` → `write`/`ls`/`read`; "`states.json`" → the episode's trace; "the Python runner exports" → the runtime exports; `new_` naming rationale kept; the unsolved branch ("## Best known approach (UNVALIDATED)") dropped because pi sends DISTIL only for a solved cell |
| `base.KEY_HYPERPARAMETERS`, `base.OUTPUT_DISCIPLINE` | `[include:key-hyperparameters]`, `[include:output-discipline]` | none |

## Guides

`guides/strict_hybrid_guide.md`, `guides/pro_hybrid_guide.md` and `guides/env_calibration.md` are RPent's files,
unchanged. RPent does not inject them either: the evaluate prompt tells the agent to read each once
(evaluate.py:413-418). They are readable (not writable) through the memory guard's `readable` roots while
`--libero-prompt rpent`; the READ THE GUIDES step maps their RPent names to pi's.

## State history (what `step` refers to)

RPent's prompts look back at earlier states (`view_env_state({"step": 0})`, `back_project(..., "step": NN)`).
Every state the model is shown becomes a state record under `<output dir>/<cell>_steps/step_NNN/`:
`state.json` (the shown result), `agentview_high.png`, `wrist_high.png`, each camera's depth
(`*_depth_high.u16.gz`, 0.1 mm units) and calibration (`*_meta.json`). `view_env_state`, `back_project`,
`segment` and `view_camera_meta` take `step` (0 = initial, -1 = latest). Each `segment` call writes
`<cell>_steps/segments/segment_NN.json` (prompt or point, camera, score, `world_xyz`), the anchors
`flash-generate.ts` reads by default. An exploration `reset` starts a new history and keeps the earlier one as
`<cell>_steps.<n>`.

A record is about 2.7 MB. `--step-history clean` (default) removes the `step_*` records when the session ends and
keeps `segments/`; `keep` keeps everything (and earlier episodes as `<cell>_steps.<n>`); `off` records nothing and
refuses look-back. Without an output dir the history is a temp dir, removed at the session's end in every mode.

## Other robots (checked against RPent eecf206)

- RoboTwin: mildly compressed. Restored in `robotwin/SYSTEM.md`: CLEAN-TO-RANDOMIZED TRANSFER (evaluate.py:39-44),
  RUNTIME (:114-122), the dropped playbook caveats (:73, :76, :78) and budget priors (:124-134), and the guide's
  robust-geometry, wrist-rotation, recovery-taxonomy and budget items (GUIDE_RPENT.md:25-28, 75-81, 118-121,
  127-130). The rest of the guide was already merged into SYSTEM.md.
- RoboCasa: faithful; restored the ENVIRONMENT sentence (evaluate.py:169-173) and "don't re-read files"
  (rpent/prompt/common.py).
- Franka, dual Franka: faithful (pi's prompts are supersets); nothing restored.
