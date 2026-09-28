# Memory
Use the LOCAL exploration corpus for this evaluation. Read every available layer relevant to the current task:
1. TASK: {{memory_dir}}/task_only/{{reference_tag}}.json and {{memory_dir}}/task_only/{{reference_tag}}_recipe.jsonl.
2. SUITE: the matching task and split entry under {{memory_dir}}/suite/.
3. GLOBAL: {{memory_dir}}/MEMORY.md, followed by only the relevant leaves under {{memory_dir}}/global/.

Treat all memory as a strategy prior. Current RGB-D, task progress, and primitive results always take precedence. Recipes provide phase order and technique, never coordinates: re-ground all xyz, xy, pixels, base poses, and fixture geometry in the current episode.[tool:rldx_skill|rldx_arm] Historical entries may name vla_act, use_prompt, or atomic prompts; use the current [tool:rldx_skill]`rldx_skill`[/tool:rldx_skill][tool:rldx_skill][tool:rldx_arm] / [/tool:rldx_arm][/tool:rldx_skill][tool:rldx_arm]`rldx_arm`[/tool:rldx_arm] tools with the complete live task language.[/tool:rldx_skill|rldx_arm] Never read {{memory_dir}}/_internal/ during evaluation. If a layer is absent, continue with the available validated layers.
