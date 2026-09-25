# Exploration
This run is MULTI-ATTEMPT EXPLORE mode. Use fresh episodes to test materially different strategies, find a successful sequence, and leave grounded memory for later evaluation runs. You are agent {{session_number}} of up to {{session_max}} on cell `{{recipe_tag}}`, with {{attempt_budget}} attempts in this session.

## Read order
Before the first robot mutation:
1. Inspect `view_env_state` (step 0) and its head image.
2. Read relevant published task, suite, and global memory under `{{memory_dir}}/`.
3. Read `{{memory_inbox}}/wip/` and `{{output_dir}}/attempts/` for notes from earlier attempts or sessions.
Fresh observations and the current task_language override historical memory. Never replay stored coordinates across episodes.

## Memory
During exploration, write working notes only below `{{memory_inbox}}/wip/`. Before each `reset`, write `{{output_dir}}/attempts/attempt_<N>_failed.json` (N continues across attempts and agents; never overwrite an existing file) with the attempt number, approach, commands and parameters tried, observed progress, bounded failure mechanism, and one meaningful change for the next attempt. Also append a concise handoff note to `{{memory_inbox}}/wip/notes.md` under `## Attempt <N>`. `reset` and an unsolved `finish` are refused until the archive exists. After success, write concise suite or global proposals directly under `{{memory_inbox}}/`. Never write directly into published memory directories.

Every proposed file must begin with parseable YAML frontmatter.

Suite proposal template:

    ---
    id: suite_robotwin_<task-name>
    scope: suite
    suite: robotwin
    regime: {{task_config}}
    task_id: {{task_name}}
    task_language: <verbatim initial task language>
    evidence:
      cells: [{{recipe_tag}}]
      attempts: <number attempted>
      solved_seeds: [{{seed}}]
      failed_seeds: []
    confidence: single-shot
    related: []
    ---

Global proposal template:

    ---
    id: global_<kind>_<short-name>
    scope: global
    kind: <primitive|perception|strategy|failure|infra>
    title: <short descriptive title>
    applies_when: <specific applicability conditions>
    evidence:
      cells: [{{recipe_tag}}]
    confidence: single-shot
    related: []
    ---

## Runtime
The tools are the only control surface. `reset` starts an ordinary fresh episode and may resample the layout. Re-run perception and rebind all geometry after every reset.

## Budget and success
Prefer in-place recovery while the episode remains recoverable; otherwise record what happened, `reset`, and change the plan. Only a fresh `eval_success: true` confirms success. An unsolved `finish` is refused while attempts remain. After native success, stop robot actions, save the audit and memory proposals, and call `finish` exactly once.

## Output
Before calling `finish`, write the final audit to `{{output_dir}}/{{recipe_tag}}.json`. Include task_name, task_config, seed, eval_success, total attempts, final_state, and successful_strategy.

When eval_success is true, successful_strategy must list in order every recipe-eligible command and its actual parameters from the successful trajectory after the final reset. Exclude failed attempts and reset itself. Re-check the recorded post-reset trajectory before writing the audit; do not invent, omit, or reorder commands. Do not claim a successful trajectory unless a recorded success step exists after the final reset.
