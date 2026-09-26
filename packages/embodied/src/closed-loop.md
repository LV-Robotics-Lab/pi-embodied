## Closed-loop rules

These hold for every robot and every mode (adapted from OpenETA's embodied closed-loop contract):

- Ground every decision in the newest camera frames and state. A newer observation outranks earlier frames, your own plan, memory and summaries.
- One motion, then look: after every motion, look at the new frame (the motion's result carries it[tool:view_env_state|render], or observe again[/tool:view_env_state|render]) before reasoning further, issuing a motion that depends on it, or finishing.
- "The tool said done", "the world changed" and "the task succeeded" are three separate judgements. A tool that returned without an error only ran; a commanded motion is not a sensed outcome. Judge the change from the images, and claim success only from explicit evidence of completion (the task's own success signal when a tool reports one).
- A motion that timed out or returned an ambiguous result has an unknown outcome. Re-observe before retrying it or issuing another motion[tool:view_env_state|render]; until you do, the next motion is refused[/tool:view_env_state|render].
- Tell infrastructure failures (a service error, a timeout) apart from task failures (a missed grasp, a wrong target). Do not repeat an unchanged call that failed deterministically; change something or report it.
- Missing or contradictory evidence calls for another observation, not a guess.
