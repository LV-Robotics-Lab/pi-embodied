# verify2 acceptance matrix

Base: origin/main (starts at 0a9fc7157, re-synced as main moves; each row records the commit it ran on).
Box: bjb2, checkout /root/autodl-tmp/pi-verify2; runs under /root/autodl-tmp/runs/verify2/<row-id>/.
One seed (0) per row, Muse planner (selfhost/muse-glimmer-30b, thinking low) unless the mode sets the planner.
Status: pass / fail (bug + owner or fix commit) / blocked (reason) / 未上机验证 (real robot; mock coverage named).
GPU policy: light sims run lock-free when GPU1 has ≥8 GB free (eval-parallel). Model servers, Isaac and cuRobo run serialized under gpu1.lock. GPU0/Muse is never touched.
Order: the RoboDojo G0.5 fp32 window (agent a6ad5cb) comes first on GPU1. Heavy rows wait behind it; light CPU/EGL rows run meanwhile.

## A. Simulator robot × mode

Task per robot (seed 0): LIBERO libero_spatial t0 · LIBERO-Pro libero_spatial_swap t0 · LIBERO-plus (first plus suite) t0 · Robosuite Lift, plus TwoArmLift for tools/code · Metaworld pick-place-v3 · Genesis cube_pick · ManiSkill PickCube-v1 per `--robot` (panda, xarm6_robotiq, widowxai), panda_stick PushT, panda_pair TwoRobotPickCube, widowx250s PutCarrotOnPlateInScene · RoboCasa (first target50 task) · RoboLab (first task) · RoboTwin beat_block_hammer · RoboDojo stack_bowls · BEHAVIOR turning_on_radio instance 0 (2026-10-05, on Isaac Sim 6.1 / OmniGibson 3.9 with the behavior61 venv and the box's data; planner selfhost/qwen3.8-27b; lanes/A7.md).

Modes (columns). `-` means not mounted for that robot, per the README table.

| id | robot | tools | units act | units plan | units plugins | code high | code low | code raw | code S4 | code-oracle | vdm | vdm-video | memory | explore | privileged | flash rec+replay | units-verify | stateless | approval std | approval reviewed | ensemble | fallback | export sharegpt | export verl-rl | flywheel+LeRobot |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A1 | libero | | | | | | | | | - | | | | | | | | | | | | | | | |
| A2 | libero-pro | | | | | | | | | - | | | | | | | | | | | | | | | |
| A3 | libero-plus | | | | | | | | | - | | | | | | | | | | | | | | | |
| A4 | robosuite (Lift, TwoArmLift) | | | | | | | | | | | | | | | - | | | | | | | | | |
| A5 | metaworld | | | | | | | | | | | | | | | - | | | | | | | | | |
| A6 | genesis | | | | | | | | | | | | | | | - | | | | | | | | | |
| A7 | behavior | pass (mode; task failure) | pass (mode; task failure) | nr | nr | pass (mode; task failure) | nr | nr | nr | nr | nr | nr | nr | nr | - | nr | nr | nr | nr | nr | nr | nr | nr | nr | - |
| A8 | maniskill panda | | | | | | | | | | | | | | | | | | | | | | | | |
| A9 | maniskill xarm6 | | | | | | | | | | | | | | | | | | | | | | | | |
| A10 | maniskill widowxai | | | | | | | | | | | | | | | | | | | | | | | | |
| A11 | maniskill panda_stick | | | | | | | | | | | | | | | | | | | | | | | | |
| A12 | maniskill panda_pair | | | | | | | | | | | | | | | | | | | | | | | | |
| A13 | maniskill widowx250s | | | | | | | | | | | | | | | | | | | | | | | | |
| A14 | robocasa | | | | | | | | | | | | | | | | | | | | | | | | |
| A15 | robolab | | | | | | | | | | | | | | | | | | | | | | | | |
| A16 | robotwin | | | | | | | | | | | | | | | | | | | | | | | | |
| A17 | robodojo | | | | | | | | | | | | | | | | | | | | | | | | |

## B. Perception / grasp / motion services (LIBERO spatial t0 s0 unless noted)

| id | item | status | session / notes |
|---|---|---|---|
| B1 | SAM3 `--detections`: detect / select_detection / reject_detection | | |
| B2 | UniDepth `--unidepth` enhance_depth (LIBERO + ManiSkill, which has no depth) | | |
| B3 | MolmoPoint `--point`: point + ground_set, one camera and multi-camera | | |
| B4 | Contact-GraspNet plan_grasp → execute_grasp | | |
| B5 | GraspGenX plan_grasp → execute_grasp | | |
| B6 | AnyPlace plan_place → execute_place (upright; resting z vs plate) | | |
| B7 | GraspNet-1B gsnet (`graspnet1b --model gsnet`) real grasp via execute_grasp | | |
| B8 | GraspNet-1B baseline | blocked: weights | |
| B9 | AnyGrasp | blocked: license (confirm the clear refusal) | |
| B10 | Metaworld + Genesis execute_grasp yaw / 20° filter | | |
| B11 | Geometry tools | | |
| B12 | align_wrist execute=true (512 vs 1024 wrist-image alignment) | | |
| B13 | suggest_grasp (with a backend, and the no-backend mode) | | |
| B14 | follow_waypoints (gripper required; a stalled leg) | | |
| B15 | IK preview pyroki (LIBERO / ManiSkill / Genesis) | | |
| B16 | IK / collision planning cuRobo: planned move_to into an obstacle names it | | |
| B17 | Scripted grasp → place height (resting z vs plate) | | |

## C. VLA / policy servers

| id | item | status | session / notes |
|---|---|---|---|
| C1 | Pi0.5 pi0_pick on LIBERO (Stage B tool mode, server up) | | |
| C2 | OpenVLA openvla_act | | |
| C3 | OpenVLA-OFT openvla_oft_act (spatial) | | |
| C4 | OFT libero_all with --unnorm-key (one act on GPU) | | |
| C5 | GR00T gr00t_act | | |
| C6 | XPolicyLab Evo-1 xpolicy_act on RoboTwin | | |
| C7 | G0.5 on RoboDojo (agent a6ad5cb runs it; fp32 window; record its result) | pass | a6ad5cb, 3846ee408 (on 0a9fc7157): stack_bowls s0 score 1, 550/800 steps, fp32, 23 chunks / 364 actions / 30 calls, GPU1 peak 25.5 GB; /root/autodl-tmp/runs/robodojo/xpolicy-g05/stack_bowls_s0/. Caveat: tokenizer is stock Qwen3.5-2B (official processor gated) |
| C8 | RLDX (RoboCasa365 server) | | |
| C9 | LingBot (RoboTwin) | | |

## D. Show-Harness

| id | item | status | session / notes |
|---|---|---|---|
| D1 | finetuned provider (a LoRA if present) | | |
| D2 | stage_control | | |
| D3 | units ablation: mcq / coords / action_ablation | | |
| D4 | point self-check | | |
| D5 | GUMI teleop + operator + /gumi-replay | | |
| D6 | real2sim datagen ManiSkill Scheme D | | |
| D7 | real2sim datagen RoboLab | | |
| D8 | train.sh standalone smoke | | |

## E. Code mode deltas (CaP-X)

| id | item | status | session / notes |
|---|---|---|---|
| E1 | ManiSkill #7 panda_pair servo in code mode | | |
| E2 | ManiSkill #8 widowx250s clipped step | | |
| E3 | ManiSkill #28 bridge 5 fps flywheel replay | | |
| E4 | #9 RoboCasa mid-run success latch | | |
| E5 | #11 oracle budget relax (one Robosuite oracle via eval.sh) | | |
| E6 | #12 preflight refusal when not root | | |

## F. Infra

| id | item | status | session / notes |
|---|---|---|---|
| F1 | dashboard: message / withdraw / manual primitive / downloads / token | | |
| F2 | /robot-check | | |
| F3 | --serve-models auto-start | | |
| F4 | Viser | | |
| F5 | human/operator provider | | |
| F6 | web tools | | |
| F7 | object memory | | |
| F8 | hardware lock (mock) | | |
| F9 | context version + budgets | | |
| F10 | LIBERO --step-history clean (step dirs deleted, segments kept) | | |
| F11 | approval standard on LIBERO with Muse through the dashboard (only high-risk motions prompt) | | |
| F12 | Stage B manifests: code.api tiers high / low / raw / privileged per robot | | |

## G. Real robots (未上机验证)

| id | robot | status | mock coverage |
|---|---|---|---|
| G1 | Franka (RLinf, Polymetis) | 未上机验证 | |
| G2 | Dual Franka | 未上机验证 | |
| G3 | Piper / dual Piper | 未上机验证 | |
| G4 | UR5e | 未上机验证 | |
