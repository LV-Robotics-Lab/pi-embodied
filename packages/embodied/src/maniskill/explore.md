# Exploration
This run is MULTI-ATTEMPT EXPLORE mode. Use fresh episodes to test materially different strategies, find a successful sequence, and leave grounded memory for later evaluation runs. You are agent {{session_number}} of up to {{session_max}} on cell `{{recipe_tag}}`, with {{attempt_budget}} attempts in this session.

## Read order
Before the first robot motion:
1. Call `view_env_state` and study both images.
2. Read relevant task, suite and global memory under `{{memory_dir}}/` (it may be empty).
3. Read `{{memory_inbox}}/wip/` and `{{output_dir}}/attempts/{{recipe_tag}}/` for notes from earlier attempts or sessions.
Fresh observations override historical memory. Never replay stored coordinates across episodes.

## Memory
During exploration, write working notes only below `{{memory_inbox}}/wip/`. Before each `reset`, write `{{output_dir}}/attempts/{{recipe_tag}}/attempt_<N>_failed.json` (N continues across attempts and agents; never overwrite an existing file) with the attempt number, approach, commands and parameters tried, observed progress, the failure mechanism, and one meaningful change for the next attempt. Also append a concise handoff note to `{{memory_inbox}}/wip/notes.md` under `## Attempt <N>`. `reset` and an unsolved `finish` are refused until the archive exists. After success, write concise suite or global proposals directly under `{{memory_inbox}}/`. Never write directly into published memory directories.

Every proposed file must begin with parseable YAML frontmatter.

Suite proposal template:

    ---
    id: suite_maniskill_<env-id>
    scope: suite
    suite: maniskill
    regime: {{scene}}
    task_id: {{env_id}}
    task_language: <verbatim task language>
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
The tools are the only control surface. `reset` restores the same seeded scene with the gripper open. Re-localize from the new images after every reset.

## Budget and success
Prefer in-place recovery while the episode remains recoverable; otherwise record what happened, `reset`, and change the plan. Only a fresh `success: true` confirms success. An unsolved `finish` is refused while attempts remain. After success, stop robot motion, save the audit and memory proposals, and call `finish` exactly once.

## Output
Before calling `finish`, write the final audit to `{{output_dir}}/{{recipe_tag}}.json` with env_id, seed, success, total attempts, final_state, and successful_strategy: every `move_delta` of the successful trajectory after the final reset, in order, with its actual parameters. Do not claim a successful trajectory unless a recorded success step exists after the final reset.
