# Memory
Before the first action, use `read` on each of these current-task files:
{{memory_files}}
The JSON/JSONL pair is reviewed seed-0 evidence; the Markdown file is task-specific exploration memory. Treat them as strategy priors, not trajectories to replay: current RGB-D, task progress and primitive results always take precedence. Historical entries may name vla_act, use_prompt or atomic prompts; these describe VLA phases only, so use `rldx_skill` / `rldx_arm` with the complete live task language. Never replay stored xyz, xy, pixels, base poses or fixture coordinates. Never read another task's memory or any global memory. If no file is listed, solve from live observations.
