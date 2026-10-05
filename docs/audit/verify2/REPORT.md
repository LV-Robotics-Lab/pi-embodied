# verify2 — box acceptance run of pi-embodied (DRAFT: the re-run column is still to fill)

**What:** every ported feature from RPent, Show-Harness, CaP-X and OpenETA, run for real on bjb2 (2× RTX 5090; GPU0 = the Muse planner, GPU1 = sims and model servers). One seed per cell. Planner: Muse (selfhost/muse-glimmer-30b, thinking low) unless the mode sets it.
**Code:** all lanes ran on 81cfbd6d9 (box /root/autodl-tmp/pi-verify2), 2026-09-29 01:30–12:00. Rows marked R changed on main since and are re-run on the latest main (last section).
**Where things are:** per-cell evidence in lanes/L1.md … L5.md and lanes/heavy.md; box runs under /root/autodl-tmp/runs/verify2/<lane>/…; MATRIX.md (rows); RERUN.md (re-run plan).
**Status words:**
- pass: the mode engaged and behaved as designed; the task need not be solved unless the row is about success.
- pm: the mode worked, the task failed.
- F: product bug (see Bugs).
- fi: Muse crash or leaked tool-call text; rerun once, the valid retry reported.
- B: blocked.
- -: n/a by design.
- nr: optional column not run.
- 未上机验证: real robot, not run on hardware.

## Summary
- Coverage: every sim robot except BEHAVIOR ran every mode it mounts, 16 configurations. BEHAVIOR is blocked: no OmniGibson dataset, no usable Isaac Sim 5.x. Every perception, grasp, VLA, Show-Harness and infra feature ran at least once.
- Tasks solved by Muse-planned modes:
  - LIBERO: tools, units plugins, code high, VDM, memory, fallback, ensemble, units-verify.
  - Robosuite: code low. Metaworld: VDM. Genesis: VDM, explore.
  - RoboCasa: OpenDrawer.
  - ManiSkill, RoboLab, RoboTwin, RoboDojo: none by Muse. Success there came only from scripted planners and policies (G0.5 on RoboDojo, score 1).
- Bugs: 37 found. 29 are already fixed on main (13 lane commits, 16 owner fixes). 8 are open or in the next landing pass.
- Muse crashed every 20–60 min until the 11:20:34 +08 restart (FP8 GEMM kernel: FlashInfer cuDNN/cublasLt → vLLM CUTLASS). None since; every affected cell was rerun once.
- Two contributors, both fixed on main: GPU0 render contexts (unpinned LIBERO EGL, RoboLab Kit) and leaked tool-call text recorded as a valid failure.
- Real robots (Franka, dual Franka, Piper, UR5e): 未上机验证, mock coverage in section G.

## A. Robot × mode
Cells use the status words above; p* = pass with the task solved. Evidence per cell is in the lane file named in each row.

A1 LIBERO spatial t0 (L1):
- p*: tools, units plugins, code high, vdm, memory, units-verify, ensemble, fallback, flywheel+LeRobot (v3.0 via flywheel04).
- p: units act, code low / raw / S4, vdm-video, explore, privileged (+ code privileged), stateless, sharegpt, verl-rl.
- F: units plan ends planner_error at the time limit.
- nr: code-oracle; the matrix wrongly marked it n/a, and LIBERO has oracles.
- Flash: covered on A2.
- approval std p (see F11); approval reviewed p in tools mode, F in units mode (L4's deadlock).

A2 LIBERO-Pro spatial_swap (L1):
- p*: tools.
- F: units act (time-limit planner_error).
- pm: code high.
- Flash record p* after 1 manual audit repair (F: audit not JSON); replay p*.
- The other columns run the same code path as A1.

A3 LIBERO-plus (L1): tools p*, units act p, code high pm; the rest same as A1.

A4 Robosuite Lift, plus TwoArmLift (L3):
- p*: code low, oracle lift_privileged, flywheel+LeRobot.
- p: tools (+TAL), units act / plan / plugins, code high (+TAL), raw, S4, vdm, vdm-video, privileged (offered), units-verify, stateless, approval std and reviewed, ensemble, fallback, sharegpt, verl-rl.
- F: TwoArmLift oracle (E5); memory runs without a corpus; explore p but writes no corpus.
- Flash: n/a.

A5 Metaworld pick-place (L3):
- p*: vdm.
- p: tools, units ×3, code low / raw / S4, vdm-video, explore, privileged, stateless, approval ×2, ensemble, fallback, exports, flywheel (no success to export).
- F: memory (no corpus).
- Not engaged: units-verify (no finish).
- -: code high (no high tier), oracle, Flash.

A6 Genesis cube_pick (L3):
- On 81cfbd6d9 every mode fails at start: env_error, env.segment declared but not served. Run with the L3 patch.
- p*: vdm, explore.
- p: tools, units ×3, code low / raw / S4, vdm-video, privileged (called), stateless, approval std, ensemble, fallback, exports, flywheel (no success).
- F: memory (no corpus).
- fi: approval reviewed (leaked calls ×3).
- Not engaged: units-verify.
- -: code high, oracle, Flash.

A7 BEHAVIOR (L3): B everywhere. No pre-sampled instances under og_dataset (the assets dir is empty); behavior61 = OmniGibson 3.9 on Isaac Sim 6.1 (unsupported), behavior45 has no OmniGibson; no Isaac Sim 5.x venv; driver 595.71.05.

A8 ManiSkill panda PickCube (L2):
- p* (scripted source): flash rec+replay, flywheel+LeRobot.
- p: tools (after the leak fix), units act / plan / plugins, code low / raw / S4, vdm, vdm-video, explore, privileged, units-verify (scripted claim), stateless, approval std and reviewed, ensemble and fallback (after the last-model fix), sharegpt (F: counts non-model sessions), verl-rl.
- Memory: p in tools mode, F in units mode (not offered).
- -: code high, oracle.

A9–A13 ManiSkill xarm6 / widowxai / panda_stick PushT / panda_pair TwoRobotPickCube / widowx250s bridge (L2):
- p: tools, units act, code low, approval std, flywheel raw + validate.
- A12: arm honoured; A13: LeRobot via E3.
- nr: the other optional columns.

A14 RoboCasa (L4):
- p*: flywheel+LeRobot (OpenDrawer, v3.0), sharegpt success export.
- p: every mounted column. Includes code high, memory (hf; local weak), explore, units-verify, ensemble, fallback.
- Flash pm: recipe replayed, but RLDX didn't reopen the drawer.
- F: approval reviewed (deadlock).

A15 RoboLab (L4):
- p: every mounted column. Memory is weak.
- F: approval reviewed (deadlock).
- B: Flash (no solved episode).
- -: code high, flywheel.

A16 RoboTwin (L4):
- p: every mounted column. Memory hf p.
- F: privileged prompt conflict (p with a nudge); approval reviewed (deadlock).
- Flash pm: the replay engaged.
- Flywheel record partial; LeRobot B (no solved episode).
- -: code high.

A17 RoboDojo (L4):
- p: every mounted column. Memory is weak.
- F: approval reviewed (deadlock).
- B: Flash, LeRobot (no solved episode).
- -: code high.

R (A):
- A1 code-oracle; A1 units plan and A2 units act (a10b948cb).
- A2 Flash record (53f7cd8ed, e40893741); Flash on the standard suites (77b1dcf9f).
- A4/A5/A6 memory and explore (6f18957bf); A6 on plain main; A6 approval reviewed (a7406a5a8).
- A8 units memory (493266425), A8 export (562b92a5c).
- A14/A16 start without RLDX/LingBot (f1081b427, 152e2f6da); A15/A17 GPU pinning (c13a8515f, 7feb5c2dc).
- The approval-reviewed deadlock and the RoboTwin privileged prompt: re-check when fixed.

## B. Perception / grasp / motion
- B1 SAM3 --detections: not run as a row. The low-tier prompt exposes detect/select/reject; short ids were verified in verify-models. R.
- B2 UniDepth:
  - ManiSkill: plumbing p, fusion F (scale_out_of_bounds; letterboxed views pass no intrinsics; L2/B2).
  - LIBERO: not in verify2 (verify-models: agentview median error 13.5 %).
  - R: 202849899, 1b32aea76.
- B3 MolmoPoint --point: not run in verify2 (verify-models: ground_set native). R: 3496660a1, plus ManiSkill.
- B4 Contact-GraspNet: p. 10 of 104 candidates, legs within 1.2 cm, lift 9.3 cm (heavy/grasp-r3). R: c83a1bbf0.
- B5 GraspGenX: grasp p (0.99, lift 9.2 cm). Place pm: the tilted AnyPlace candidate stalled 7.7 cm short, reported correctly. R: c83a1bbf0, 20944b0ac.
- B6 AnyPlace: p for the chain. The bowl was set down, not dropped (+1.3 cm over its table z), but 8.3 cm off the plate centre: AnyPlace's own target was 4.4 cm off, on the rim. R: 20944b0ac, 92245e333.
- B7 GSNet: pm. Realsense: 0–1 on-object candidates; kinect: 4. The top candidates approach sideways (pitch −1.3 to −1.5), so execute_grasp stalls or misses. R: c83a1bbf0.
- B8 GraspNet-1B baseline: B (weights only on Drive / Baidu).
- B9 AnyGrasp: p, refused with "AnyGrasp license file missing".
- B10 Metaworld/Genesis 20° filter:
  - p: every candidate off by 25–73° refused unmoved.
  - F: next_after hit a stale id (no state digest).
  - R: L3 23ec17fa0 (next pass).
- B11 geometry tools: exercised in A14 tools only. R as an explicit row.
- B12 align_wrist: p, 1024 px consistent, residual 163 → 72 px after move_to. No execute parameter on 81cfbd6d9. R: OpenETA wt3.
- B13 suggest_grasp: F (by design on 81cfbd6d9): refuses to start without a plan_grasp backend; the with-backend case was not run. R: OpenETA wt3.
- B14 follow_waypoints: p (stalled leg → not_reached; >0.3 m refused). The gripper is not required. R: 2138785e4 + OpenETA wt3.
- B15 PyRoKi IK preview: p on ManiSkill panda/xarm6 and Genesis; LIBERO passed 9/9 in verify-models.
- B16 cuRobo: F. The move was refused unmoved, but only "IK_FAIL" with nearest=null (heavy/curobo/probe.out). R: b11631997, 28545a067.
- B17 place height: p (set down; see B6). R with B6.

## C. VLA and policy servers
- C1 Pi0.5 pi0_pick: p*, 3 calls, success (L1/A1/tools). R: 60e725d0d (LIBERO starts without Pi0.5).
- C2 OpenVLA: pm. 4 calls, 0 judged successful, 735 steps; the previous round solved this cell (heavy/vla/ep-openvla).
- C3 OpenVLA-OFT spatial: p*, success at step 591, 0.155 s per chunk.
- C4 OFT libero_all --unnorm-key: p, 0.151 s per chunk, deterministic. F (minor): a local copy reports suite None even with --suite.
- C5 GR00T: pm. 4 calls (1 pick success), 1681 steps, turn budget; 0.12 s per chunk.
- C6 Evo-1 xpolicy_act on RoboTwin: p, 400 actions (L4/C6).
- C7 G0.5 on RoboDojo fp32: p*, score 1, 550/800 steps. Caveat: stock Qwen3.5-2B tokenizer.
- C8 RLDX: p (L4/A14). R: f1081b427.
- C9 LingBot: p (L4/A16). R: 152e2f6da.

## D. Show-Harness
- D1 fine-tuned v5 LoRA: pm. 200 steps, 0 fallbacks, but off-distribution (MV_RIGHT ×124, end effector at x = −1.42). R: bcaa5b79f; the sh-modes v5view rerun result is still to collect.
- D2 stage_control: p.
- D3 mcq / coords / ablation blind and bare: p ×4.
- D4 point + self-check: p.
- D5 GUMI + operator + /gumi-replay: p.
- D6 real2sim ManiSkill Scheme D: p.
- D7 real2sim RoboLab: RubiksCube p (2/2, 1/1, 1/1); BananaInBowl F (0/8). R: 2a634d714, 7feb5c2dc.
- D8 train.sh smoke: p (loss 4.12 → 0.61). The no-rollout traceback is fixed in a3bbad597.

## E. Code-mode deltas
- E1 #7 panda_pair servo: p.
- E2 #8 widowx250s clipped step: p.
- E3 #28 bridge 5 fps flywheel → LeRobot: p.
- E4 #9 RoboCasa mid-run latch: F, not implemented. R: 5c461e32a.
- E5 #11 oracle budget relax: F (TwoArmLift died at the 3 m cap). R: 3577387ec.
- E6 #12 non-root refusal: p (uid 65534). R: 5b465f4fe (refused at start via code.preflight).

## F. Infra
- p: F1 dashboard, F2 /robot-check, F3 --serve-models, F4 Viser, F6 web tools, F7 object memory, F8 hardware lock, F9 budgets and context version.
- F5 human/operator: F, then fixed: every operator tool call failed under rpc/json. R: 7263af018.
- F10 --step-history: B (not on 81cfbd6d9). R: 5bf9d6d15.
- F11 approval standard: as implemented (no risk tier; the dashboard can't answer a confirm). R: approval2.
- F12 manifests / tiers: p on LIBERO, and every robot via its code columns. R: 86ad2e2b7, 3e66dd132 (S3 ≠ S4 everywhere).

## G. Real robots (未上机验证)
- G1 Franka: franka.test.ts 5, test_franka_code.py 21, test_franka_polymetis.py 46, hardware lock, operator tools. Not covered: a real RLinf worker, libfranka/Polymetis, cameras.
- G2 dual Franka: dual_franka.test.ts 7, per-arm code, operator and lock tests. Not covered: a real two-arm deployment.
- G3 Piper: piper.test.ts 24, test_piper.py 44 (the workspace box clamps, unlike Franka/UR5e), test_piper_code.py 11, L5 F8 (a held arm refused before ROS). Not covered: CAN/piper_sdk, ROS, cameras.
- G4 UR5e: ur5e.test.ts 13, test_ur5e.py 59, test_ur5e_code.py 15, lock, flywheel rules. Not covered: ur_rtde, Robotiq, cameras.

## Bugs
### Fixed on main (landed ids)
1. time-limit → planner_error: a10b948cb.
2. audit not JSON: e40893741, 53f7cd8ed.
3. Flash only Pro suites: 77b1dcf9f.
4. leaked tool call counted as a valid failure: a7406a5a8.
5. oracle hint: 79e5ffb3a.
6. ensemble/fallback use the first --model: abcc971c6.
7. human/operator in RPC: 7263af018.
8. DONE WHEN ×2: d45738489.
9. train.sh traceback: a3bbad597.
10. E4 mid-run latch: 5c461e32a.
11. E5 budgets: 3577387ec (L3's ac54ff407 is a duplicate).
12. cuRobo obstacle name: b11631997, 28545a067.
13. step-history: 5bf9d6d15.
14. GPU0 leakage (LIBERO EGL, RoboLab Kit): c13a8515f, 7feb5c2dc, 422394b2d.
15. explore/memory corpus on Robosuite/Metaworld/Genesis: 6f18957bf.
16. UniDepth intrinsics on ManiSkill: 202849899.
17. dead perception server offered: 1b32aea76.
18. skill servers required at start: 152e2f6da, f1081b427, 60e725d0d.
19. replay with no plan counted as a valid failure: 552bd78fa.
20. memory missing in units/code mode: 493266425.
21. export counts non-model sessions: 562b92a5c.
22. README table: cd008708f.
23. flywheel04 venv: 0c4da006e.
24. place off the region: 20944b0ac, 92245e333.
25. tilted candidates: c83a1bbf0.
26. v5 4:3 crop: bcaa5b79f.
27. BananaInBowl: 2a634d714.
28. Genesis env.segment: on main (L3's 36b35de1d superseded).
29. S3 = S4 prompts: 86ad2e2b7, 3e66dd132.

### Open / next landing pass
30. approval reviewed + units deadlock (images:0, latestImages reads toolResults only), on 4 robots. Owner: capabilities/operator.ts.
31. RoboTwin SYSTEM.md forbids poses, which contradicts --privileged. Still on main.
32. Metaworld/Genesis grasp ids expire on a refused execute_grasp. L3 23ec17fa0, next pass.
33. approval standard has no high-risk tier; the dashboard can't answer a confirm. approval2, next pass.
34. suggest_grasp no-backend, follow_waypoints required gripper, align_wrist execute. OpenETA wt3, next pass.
35. OFT local copy reports suite None, so the libero_90 refusal is bypassed. Owner: openvla_oft_server.
36. Raw Python tracebacks reach the model (over-long move, below-threshold segmentation). Open; 1b32aea76 covers only a dead server.
37. Weak local memory on RoboLab/RoboDojo: Muse doesn't read the seeded dir. Planner compliance.

Not a bug: rldx_skill, lingbot_act and pi0 are not in run_code, by design (5a7fb1a89).

## Infra notes
- Muse stabilization: 11:20:34 +08 (see Summary). Crash tally before it: L1 ~6, L2 ~15 plus 7 leaked-call replies, L3 ~6, L4 7, L5 3. No invalid cell remains.
- GPU0 leakage: bug 14. ManiSkill SAPIEN keeps a harmless 6 MiB Vulkan enumeration context on GPU0.
- GPU1 scheduling: from 02:44, /root/autodl-tmp/tools/gpu1-queue.sh (named queue + flock) with a keeper. Lock-free jobs keep ≥16 GB free; locked batches ≤30–45 min; servers stopped before release. It replaced flock races with 6 waiters for over an hour.
- Test flakes (root cause, fixed in 1e27c8b4a): services/tests/humanclaw_golden.py generate() set planner_module.time.sleep = lambda s: None. That is the global time module, never restored, so every later test in a full-suite run slept zero seconds. Hence the rpc_facade stop/healthz (and camera restart) failures that happened only in full suites, deterministically, never alone. 1e27c8b4a restores it, adds a conftest guard that fails any test leaving time.sleep/monotonic/time replaced, and judges the rpc_facade tests server-side. 4/4 failures before, 900 passed ×2 after under the same load.

## Re-run on the latest main (bc9697b3, 2026-10-05 18:48-20:13 +08)

One fresh box checkout of main bc9697b3 (HumanCLAW audit fixes + docs), synced by box-check.sh (npm run check ok, node tests 844 pass / 0 fail, services/test.sh green) and run with the migrated flag surface: --deployment bjb2 plus a project .pi/embodied.json for verify3's servers, --env-url/--task/--arm/--aux-model, hand-driven rows through the dashboard's POST /primitive from a --mode rpc client. Evidence per row: lanes/RERUN-RESULTS.md.

Planner: Muse is gone; selfhost/qwen3.8-27b (NInfer, GPU0, thinking low). Success numbers are not comparable to 09-29; rows are judged by "mode engaged and behaved as designed". Qwen solved cells Muse never did (Robosuite Lift, Metaworld pick-place and Genesis cube_pick in exploration; LIBERO spatial t0 / spatial_swap t0 in tools mode and under --approval standard; the object_swap_7 privileged oracle). Qwen leaked no tool call as text (0 of ~60 sessions); its failure mode is NInfer's serial queue timeout (planner_error, retried; 4 hits on one cell). GPU1: the user's Pi0.5 and SAM3 reused and never restarted; verify3's servers under gpu1.lock; Isaac rows left GPU0's compute list untouched.

| row | 81cfbd6d9 | bc9697b3 |
|---|---|---|
| A1 units plan / A2 units act at the time limit | F | pass |
| A1 code-oracle high / privileged | nr | pm / pass (solved) |
| A2 Flash record (write_audit) | F | pass (+ bug 42: flash-generate needs --language) |
| A1 Flash record, standard suite | - | pass |
| A2 / A1 Flash replay | pass / - | pm / pm (needs Molmo: bug 43) |
| A4 / A6 explore -> corpus -> memory | F | pass; A5 pm (budget / unsolved) |
| memory without a corpus refuses | F | F (bug 38) |
| A6 plain main; approval reviewed tools / units | fi | pass / pass / pass (bug 30 fixed, also LIBERO units) |
| A8 units memory; A8 export | F / F | pass / pass |
| A14 / A16 without RLDX / LingBot; --require-skills | F | pass |
| A16 --privileged without a nudge (bug 31) | F | pass |
| A15 / A17 GPU pinning | F | pass / pass (A17 on a shadow RoboDojo root) |
| B1 detections | nr | pass |
| B2 UniDepth LIBERO / ManiSkill | - / F | pass / F (bugs 16, 40) |
| B3 MolmoPoint one camera / set | nr | pass with Molmo2 (one camera) / B (MolmoPoint OOM) |
| B4 + B6 + B17 CGN + AnyPlace chain | pass / pm / pm | pass (0.8 cm from the plate centre) |
| B5 GraspGenX | pm | pm (+ finding 44) |
| B7 GSNet | pm | pass (plan-time tilt filter) |
| B10 next_after | F | pass |
| B11 geometry; B12 align_wrist execute; B13 no backend; B14 waypoints | nr / pass / F / pass | pass / pass / pass / pass |
| B16 cuRobo refusal names the obstacle | F | F (bug 45: nothing blocked) |
| C1 without Pi0.5; C8 / C9 optional skills | - / F | pass / pass |
| D1 v5 views | pm | B (GPU memory) |
| D7 RoboLab BananaInBowl real2sim | F | pass (2/2) |
| E4 mid-run latch; E5 budget relax; E6 preflight | F / F / pass | pass / pass / pass |
| F5 human/operator; F10 step-history; F11 approval standard; F12 S3 vs S4 | F / B / as impl. / pass | pass / pass / pass (+ bug 41) / pass |
| bug 35 OFT suite | F | B (GPU memory) |
| bug 36 tracebacks | F | pass |
| Muse malformed call -> planner_error | - | nr (unit test only) |

Box: three runtime files still carry pre-migration paths (~/.liberopro + ~/.libero config.yaml, the AnyPlace pi eval yaml, RoboDojo's Assets/Robots/x5/curobo.yml); LIBERO, AnyPlace and RoboDojo cannot start on the box until they are regenerated. verify3 used corrected private copies; none of the user's files were edited.

### New bugs from the re-run (evidence in lanes/RERUN-RESULTS.md)
38. Memory without a corpus does not refuse: capabilities/memory/index.ts throws in its session_start handler; pi logs "Extension error" and the episode runs memory-less, recorded as a valid failure. Needs the robot's fail-closed start.
39. Ambiguous deployment is silent: with several deployments and no default, config.ts select() returns an empty deployment and no configProblem; python.* unresolved -> env server on PATH's python, recorded as env_error without the cause.
40. ManiSkill wrist view hands UniDepth invalid intrinsics ("K must have positive focal lengths"); bug 16 still open on agentview (used_intrinsics true yet scale_out_of_bounds).
41. Approval counters: operator.ts ~629-632, the else binds to the run_code if, so approved non-run_code motions also increment approval_rejected.
42. write_audit audit lacks task_language, so flash-generate fails ("pass --language") on every recorded cell.
43. --flash-reanchor=false is ignored: pi sets boolean extension flags true whatever the value (agent-session-services.ts:106); a replay without a Molmo server fails closed; the migration doc names it as the replacement for --molmo off.
44. plan_place with held=false still returns placements and execute_place runs them with an empty gripper (nudged bowl_1 by 3 cm); on 81cfbd6d9 it refused.
45. cuRobo --ik blocks nothing on LIBERO: targets in the bowl wall, 3 cm below the table top and inside the plate are "planned" and executed to a physical stall; on 81cfbd6d9 such targets were refused (IK_FAIL). Suspect 02de8be8's table carve-out or the collision world sent by the server.

Blocked by GPU memory beside the user's 16 GB of servers: D1, bug 35 (OFT), MolmoPoint-8B (B3 set). Not reproducible with Qwen: the leaked-call row (unit test only).

### Status of the re-run bugs (2026-10-06)
All fixed on main: 38 → 61cd1150, 39 → 96d77926, 40 → 96d50540 (bug 16 on agentview is a UniDepth model limitation; the scale_out_of_bounds refusal is correct), 41 → f616acd4, 42 → c5a22756, 43 → 61c03b24 (`--flash-reanchor on|off`), 44 → 8b1762c1, 45 → d82f102a (root cause: motion.py's 4 cm near-drop removed the table itself when the goal lay inside it; fixtures are now fixed obstacles and a goal >5 mm inside one is refused before planning; verified live on the box with cuRobo). Codex review findings on the same range: hardware lock keeps address + serial locks (fccfa156), approval_rejected counts refusals only (f616acd4), explicitly given default-valued flags are given (06a85fcf).
