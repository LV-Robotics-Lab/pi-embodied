# 交接：审计遗留修复 + 四仓库迁移补全（main @ d3b79ff，1.9 节基于 a2c880c）

**仓库**：LV-Robotics-Lab/pi-embodied。直接在 main 上改、提交、推送（用户已授权）。

**规矩**（详见根目录 AGENTS.md）：
- **检查**：改完代码跑 `npm run check`（看完整输出），再跑 `ruff check` 和 `ruff format`。
- **测试**：
  - TS：在 `packages/embodied` 下跑 `node --test test/<f>.test.ts`；
  - Python：在 `services/` 下跑 pytest。
  - 注意：本环境跑 TS 测试要先编好各包的 dist。见下面第一条注意事项。
- **git**：只 `git add` 自己改的文件；提交信息的格式是 `fix(embodied): ...` 或 `feat(embodied): ...`。
- **工具 schema 变了**：用 `UPDATE_TOOL_SCHEMAS=1` 重新生成 `test/fixtures/tool-schemas.json`，并确认 diff 只包含预期的改动。
- **已知的偶发失败**：`eval-parallel` 的 "interrupted run" 测试会偶发失败，不算回归。

**依据**（都在审计 session 的 scratchpad 里，用户可以转给你）：
- `audit-f31a615.md`：对上一轮 11 个修复提交的审计，条目编号下文沿用；
- `migration-coverage-f31a615.md`：四个源仓库的迁移覆盖对照；
- `audit-a2c880c.md`：最新一轮（135 个提交）的审计，1.9 节依据它。

**源码克隆**（不在本仓库，要用时请用户提供或自己 clone 到临时目录）：
- RPent eecf206
- Show-Harness 137d571
- CaP-X 53e9966
- OpenETA 7d4a0a1 + openeta-for-codex 分支
- ur_rtde 1.6.5 源码
- librealsense v2.54.1 源码

之前的 `handoff-xpolicylab-robodojo.md`（XPolicyLab 客户端、RoboDojo）仍然有效，还没开始做，放在本文第 4 阶段之后。

**本环境的两个注意事项**：
1. TS 测试需要上游各包的 dist。按依赖顺序在每个包目录下跑 `npx tsgo -p tsconfig.build.json`：
   - 顺序：chord、tui、telemetry、ai、durable、agent、session-backends/sqlite-node、protocol、client、server；
   - `ai` 编完后要把 `src/providers/data` 拷到 `dist/`；
   - coding-agent 用 `npm run build:unbundled`；
   - 不要跑 `npm run build`。
2. 不能上 GPU 或真机的部分，要在提交说明里写"未上机验证"。

---

## 第 1 阶段：修已迁功能里坏掉的（先做，每条单独提交）

### 1.1 OpenVLA 在所有 LIBERO-Pro 套件上被拒（audit #1，中高）
- **现象**：服务端 `--suite libero_10` 时，每次 `openvla_act` 都被拒。
- **原因**：`packages/embodied/src/vla-adapters.ts:109` 的 `suiteMismatch` 做精确比较，但 `robot.task.suite` 通常是 `libero_10_task`、`libero_object_swap` 这类 Pro 变体。
- **怎么改**：
  - 先把后缀去掉，得到基础套件再比较。后缀以 LIBERO-Pro 实际用的为准（`_swap|_task|_lan|_object|_temp` 等，读 `libero/index.ts` 里 LIBERO_TYPE=pro 的处理确认）。
  - 修正报错文案：`openvla_server.py` 的 `--suite` 选项只有 4 个基础套件，不能提示用户传 `--suite libero_10_task`。
- **顺手补（audit #19）**：
  - OFT 和 GR00T 的 facade 把 suite 传进 `vla.info`。GR00T 的默认权重 `n1.6-libero-spatial` 只适用于 spatial。
  - 修正注释"`libero_all` 覆盖所有套件"：`libero_all` 不含 libero_90。
- **测试**：`test/vla-adapters.test.ts` 加 Pro 套件的用例。

### 1.2 抓取链不通（audit #20、#21，中高）
**(a) LIBERO：同一个 grasp_id 的第二次 `move_to` 必然被拒**
- 文档（`libero/SYSTEM.md:37`）要求先 `standoff 0.10` 走到预抓取位，再 `standoff 0` 下去。但每次 `move_to` 都会调 `env.resolve_grasp`（`libero/index.ts:613`），而第一次移动的 `env.step` 已经让 id 作废。
- `plan_place` 返回的 p id 也在抓取的第一步就作废，所以"`move_to` 到 p id"不可能成功。
- **推荐做法**：在 TS 端缓存。第一次 `resolve_grasp` 时一次性取回：
  - 候选的世界位姿；
  - 接近方向；
  - 这个 id 对应的所有 standoff 所需信息。

  之后用同一个 id 调 `move_to` / `move_pose`，只要确认"id 是这个 episode 里规划的，而且期间没有 reset"，就直接用缓存的位姿。
- **p id**：`plan_place` 的结果同样缓存。放置时物体在手里，位姿以世界坐标为准，不受运动影响。
- **SYSTEM.md 要写清楚**：
  - 缓存的位姿是规划时的世界坐标；
  - 物体动了（被推开、抓空）就要重新规划。
- **Franka 和 dual Franka 不受影响**：它们的工具不接收 grasp_id。
- **测试**：LIBERO 的 TS 测试模拟 `plan_grasp` → `move_to(id, 0.10)` → `move_to(id, 0)` 成功，reset 之后同一个 id 被拒。

**(b) Franka 和 dual Franka 的规划器拿不到 SAM3**
- **Franka**：`robots/franka/env_server.py:159-166` 的 `_grasp_planner` 没有把 sam3 URL 传给 `GraspPlanner.from_args`。结果是 `plan_grasp(object=文字)` 和 `plan_place(region=文字)` 总是失败。
- **dual Franka**（`robots/dual_franka/env_server.py:96-102`）有三处要改：
  - 同样要传 sam3；
  - TS 端（`dual_franka/index.ts:648-652`）没把 `--robot-sam3` 传给服务端；
  - 感知模块的相机名（`wrist` / `third_person`）和规划器的相机名对不上，`_mask_for` 会拒。需要统一相机名，或者加一层映射。
- **测试**：在 `services/tests/` 加用例，用 fake backend 驱动 `FrankaEnvFacade` 走完"文字 → plan_grasp → plan_place"。审计 agent 在 `scratchpad/audit2/test_audit.py` 里写过类似的脚本，可以向用户要来参考。

**(c) Franka 的 polymetis 后端会收到 `--graspnet` 等参数（audit #22，低，未实测）**
- `franka/index.ts:1207` 会给 polymetis 后端传 `graspArgs(pi)`，但 `franka_polymetis/env_server.py:611-639` 的 argparse 里没有这些参数。
- **二选一**：polymetis 不传这些参数，或者 polymetis 也接上抓取规划器。

**(d) 两个小项**
- LIBERO 的 `env.rotate_wrist` 没被运动包装（audit #23）。把它加进 `grasp.install` 的 `mutating` 列表，也就是 `libero/env_server.py:325` 附近。
- 客户端就拒掉的 Franka 运动仍然会取观测，导致 id 作废（audit #6）。改成客户端被拒时不调用 `dumpState`。

### 1.3 代码模式沙箱（audit #7–#13）
按以下顺序改，全部在 `services/pi_embodied_services/utils/code_exec.py`（`RpcFacade` 在 `utils/rpc/`）：

1. **防止子进程读服务端内存和环境变量（#7，高）**
   - 服务端进程启动时调用 `prctl(PR_SET_DUMPABLE, 0)`（ctypes 调 libc）。之后同一 uid 的非 root 进程就读不了 `/proc/<pid>/environ` 和 `/proc/<pid>/mem`。
   - 如果以 root 运行，这一条挡不住。可选的加强：子进程 setuid 到 nobody，或者启用 seccomp。
   - 修改 docstring 和 README 里"已清掉秘密"的说法：要说明凭据文件仍然可读、root 下无效。
   - 测试：以非 root 身份启动时，子进程读 `/proc/{ppid}/environ` 失败。

2. **程序经本地 RPC 端口绕过预算（#8，中高）**
   - `RpcFacade` 在 `code.run` 执行期间，对到达的业务调用直接报错，不再排队。放行的例外：`stop`、`healthz`，以及 `code.run` 自己经 registry 发起的内部调用（它们不走 HTTP，本来就不受影响）。
   - 实现方式：在 `_run_call` 里检查一个"code run 正在执行"的标记。
   - 测试：运行期间用 HTTP POST `env.step` 会收到错误；运行结束后的调用正常。

3. **NaN 让移动上限失效（#9，中）**
   - 反序列化时加 `json.loads(..., parse_constant=...)`，遇到 NaN 或 Infinity 直接拒绝。
   - `_call` 里如果 `move` 不是有限值，就拒绝这次调用。
   - 同时检查 LIBERO 的 `move_delta` / `move_to` 本身对 NaN 的检查（`np.linalg.norm(nan) > x` 为 False）。

4. **畸形消息让整个 `code.run` 抛异常（#10，中）**
   - `_serve` 捕获所有 `Exception`（包括 RecursionError 和 TypeError），按 "died" 处理，保证 `_finish` 一定会被调用。
   - 原语返回了编码不了的值时，给子进程回 error，不要让父进程抛异常。

5. **root 下孙进程能活过本次运行（#11）**
   - 在进程组之外再加一层兜底：运行结束时扫 `/proc`，找出 session id 等于子进程 pid 的进程并 kill；或者用 cgroup。
   - 至少要改文档，写明 root 下这个限制无效。

6. **清理环境变量的竞态窗口（#12）**
   - 改用 `subprocess.Popen(env=scrubbed, pass_fds=[fd])` 启动一个小的引导模块，再把 fd 包成 `multiprocessing.connection.Connection`。
   - 这样就不用改 `os.environ` 了。

7. **不响应 stop 的原语让墙钟超时失效（#13，疑似）**
   - 给 `segment`、`plan_grasp`、`plan_place` 在代码模式下的超时设上限，取本次运行剩余时间和它们原来超时中较小的那个。
   - 零位移的 `chunk_step` 加步数上限，并在每步之间检查 stop。

8. **TS 中止路径（#4）**，改 `code/index.ts`：
   - 中止时如果 `code.run` 还没发出去，就不再发送。可以在 `RpcClient` 上加一个"仅在未发送时取消"的选项，或者先检查 busy 队列。
   - LIBERO 的 `code.observe` 在中止后取 `env.raw_obs` 不要带已中止的 signal，改用新的 signal 或者不传。
   - 服务端 `run()` 开头清 `_abort` 之前，要先检查有没有待处理的 stop。

### 1.4 相机和 UR5e（audit #3、#5、#14–#18）
UR5e 没有硬件，全部用 mock 测。凡是依赖真机行为的，提交说明里都要写"未上机验证"。

1. **`inverse_brown_conrady` 去畸变公式过时（#3，中）**
   - 位置：`components/cameras/base.py:171-172`。
   - 改成和 `modified_brown_conrady` 一样的迭代求逆，与 librealsense v2.54.1 `rs.cpp:3512-3601` 一致。
   - 更新 `test_cameras.py:119` 和 `test_ur5e.py:1167`。
2. **旧标定目录加载不了（#5）**
   - 读取 `intrinsics.json` 时，兼容去掉 `distortion.` 前缀。
3. **异步运动可能刚发出就判定"已结束"（#14，疑似高）**
   - 位置：`robots/ur5e/control.py:374` 和 `:622`。
   - 在 `move_l` / `move_j` 之后，先等 `getAsyncOperationProgressEx()` 的操作 id 变化（设短超时），确认运动已经开始，再轮询是否结束。做法参考 ur_rtde 自带的 `examples/py/move_path_async_example.py`。
   - `hardware.py` 要暴露 progress-ex。
   - mock 要能模拟"启动有延迟"的情况，并加一个测试。
   - UR5e facade 还要覆盖 `_on_stop`，保证在 RPC 之外也能停下机械臂。
4. **运动失败识别不全（#15）**
   - 轮询循环里检查 `safety_stopped()`；一旦检测到，立刻退出并报告，而不是空转到超时。
   - 目标不可达时（脚本停止、程序没有运行）尝试 `reuploadScript`，或者报错说明需要重启。
   - 修正 `control.py:20-21` 的注释，它夸大了返回值的作用。
5. **reset 不检查路径和起始位（#16）**
   - 运动前先用 `getForwardKinematics(begin_joints)` 检查起始位是否在工作空间内。
   - 抬起那一步走 `check_target`。
   - 抬起的 moveL 被拒时，保留夹爪结果。
6. **`max_frame_age_s` 对真实相机不生效（#17）**
   - RealSense 改用 `color.get_timestamp()`，webcam 改用 `CAP_PROP_POS_MSEC`，换算成 monotonic 时间来算帧龄。换算方法写进注释。
7. **小项（#18）**：
   - `example.yaml:6-7` 的说明和代码相反，改过来；
   - TS 确认框里写明"抬起并松开持有物体"；
   - `info.note` 改成列表，不要互相覆盖；
   - 修 RTSP `close()` 和读线程的竞态；
   - webcam 读取加总超时上限；
   - `empty_width_m` 核对 2F-85 空抓的实际宽度（约 9 mm），必要时调整默认值。

### 1.5 Flash 的转向重放（audit #2，中）
- **位置**：`packages/embodied/src/flash/generate.ts:124-133`。
- **拆段**：单条结果的航向变化拆成若干段，每段不超过 0.3 rad（RoboLab 的 `MAX_ROTATE_RAD`，最好从机器人那边读，不要写死）。
- **回绕**：`turned` 折回到 (-π, π]。
- **测试**：在 `flash-recipe.test.ts` 里加两个用例：`n=3` 的 ROTATE_CW（0.45 rad）拆成 2 段；从 179° 转到 -179° 算作 +2°。

### 1.6 最新两个提交的审计结果（依据：`audit-7766772.md`）

**310dce4（ManiSkill 多机械臂）**

1. **xarm6 做 PullCubeTool 过于容易**
   - 现象：约 8% 的种子一 reset 就算成功。
   - 原因：成功条件是"方块离机器人底座 < 0.6 m"，而 xArm6 底座在 x=-0.522，比 Panda（x=-0.615）近。
   - 修法：从 xarm6 的白名单里去掉这个任务，py 和 TS 两处都要改；或者拒绝开局就成功的 episode。
2. **xarm6 在代码模式下，原始 `step` / `chunk_step` 的夹爪符号是反的**
   - 修法：服务端统一映射夹爪符号；或者在代码 API 里暴露 `gripper_action`。
   - 同时修正 `maniskill/index.ts` 约 440 行过时的注释。
3. **顺手修的小项**：
   - `add_ee_control` 的幂等标记读错了对象：`env_server.py` 约 329 行读的是 property，但标记设在 `prop.fget` 上。
   - `eval-parallel.sh` 的汇总标签没带机械臂型号。
   - `maniskill -` 配 `--robot xarm6_robotiq` 会默认选到 BlockPAP rig，导致每次都报错。
   - StackCube / PlaceSphere 在 xarm6 上的 dense reward 可能除以零：先按 xarm6 的关节顺序核实。

**7766772（没有腕部相机时关掉插件）**

1. **Piper 没有腕部相机时提示词自相矛盾**
   - 原因：`piperViews`（`piper/index.ts:146-172`）没有跟随 `wrist()`。
   - 修法：让它跟随；并且在没有腕部相机时拒绝 `--view-select`。
2. **UR5e 的固定相机没写 `mount` 时被当成腕部相机**
   - 修法：服务端 meta 公开由标定得出的 `eye_on_hand`，TS 端以它为准。
3. **fine-tuned 在没有腕部相机的机器人上**
   - 只有一张图时，第一步就报错。
   - 两个都是固定相机时，第二张会被当成腕部图。
   - 修法：启动时就检查相机数量和类型；不符合时拒绝，或者要求显式传 `--ft-cameras`。
4. **小项**：
   - 显式要求开的腕部插件在没有腕部相机时要直接报错，和 `--units-rt` 保持一致。
   - 修正 mem_text 里"两个视角"的措辞。
   - 快照测试补上 dual Franka 有腕部相机的变体。
   - 给 Piper、UR5e、dual Franka 的 `wrist()` 逻辑补测试。
   - 修正 7766772 提交说明里对 dual Franka 的描述：默认配置实际失去了 variable_step、action_chunk 和旋转补偿。只能在后续提交说明里补充，不要改历史。

### 1.7 7e0bf0d 审计出的新问题（依据：`audit-7e0bf0d.md`，按顺序做）

**先修高危**：
1. **沙箱**：服务端非 root 运行、又没配沙箱 uid 时，拒绝代码模式；或者让 pi 启动时也设置 `PR_SET_DUMPABLE=0`。二选一，并修正 docstring。
2. **`/gumi-replay`**：
   - 任何一步 `{ok:false}`、stop 或 Interrupt 都要中止回放；
   - 真机回放前先确认；
   - 给 `--pause` 一个非零默认值；
   - 补测试，覆盖失败和中止两种情况。
3. **`/robot-check`**：`ENV_CALLS` 改成 `env.get_env_meta`；测试里的 mock 只接受真实的方法名。
4. **planner 导出**：
   - 图像列表和 `<image>` 占位符在裁剪之后一起生成，保证数量一致；
   - VeRL 的 RL 行里，`prompt` 和 `images` 必须对应；reward 改存可复现的 env seed（参考 CaP-X）；
   - VeRL 的 SFT 图像写成 `{"image": 绝对路径}`。

**再修中危**：
1. **Franka 规划器的快照**：每次观测都要清快照，或者把帧签名纳入"是否移动"的判断。scene reset 之后必须让旧 id 失效。
2. **`execute_place` 失败时的"手里有东西"记录**：只有在张开夹爪完成、或者确认手里已经没东西之后，才能清掉。
3. **UR5e 重连后的异步等待**：先等寄存器变成 (0, 未运行)，再读 `before`。mock 在重传时要清 op_id，并补一条测试。
4. **UR5e 零 TCP 偏移时的 FK**：把零偏移换成一个极小的非零值（强制走 RECIPE_12），或者在本地计算 FK；同时修正 `hardware.py` 的 docstring。
5. **执行抓取时的姿态**：伺服完整姿态，包括 roll 和任意方向的倾斜；做不到的话，拒绝接近方向不能表达的候选。
6. **Franka 系列在 facade 进程里的 `--robot-config`**：调用 `set_robot_config_path`，让规划器和 `perception_layout` 读用户自己的配置。
7. **`plan_place` 的描述要按机器人区分**：Franka / dual Franka 不接 `claim_waypoints`，就不要在描述里写"抓取后放置"这条流程。
8. **dashboard `/primitive`**：
   - 走 `takeover.humanStep`，并写入会话条目；
   - 更新 GUMI 的观测和机器人状态；
   - `/message` 在手动调用执行期间要拒绝，或者排队；
   - 共享的 signal 不能被覆盖。
9. **dual Franka 手动调用器**：
   - 工作空间和地板限位改到服务端检查；
   - Ctrl-C 时发送 `env.stop`；
   - `--max-move` / `--max-rotate` 加上限。
10. **标定采集 `--write`**：先显示旧值和新值，要求确认，并写一份 `.bak` 备份。
11. **5f868b2 补文档**：给 `packages/coding-agent/CHANGELOG.md` 和 `packages/agent/CHANGELOG.md` 的 `[Unreleased]` 补上条目，并更新 `docs/extensions.md`。
12. **导出记录的系统提示词**：
    - `session_start` 时重置 `recordedPrompt`；
    - 在 explore 改写完提示词之后再记录。
13. **导出的过滤**：默认排除 `--privileged` 的运行、operator 判定的运行，以及 explore 在 reset 之前的尝试；分别提供开关，允许显式包含。

**ManiSkill 多机械臂 5 条仍未修**：见 1.6，先于低危项处理。

**低危项**：按 `audit-7e0bf0d.md` 第四节顺手修。

### 1.8 b1a9e9d 审计出的新问题（依据：`audit-b1a9e9d.md`）

**先说顺序**：1.7 的高危项到现在一个都没修。上一轮做的是第 4–7 阶段的内容。**请先回头做完 1.7，再做本节，然后再继续第 2 阶段以后的内容。**

**高**
1. **RoboDojo 的 flywheel 要按每一步写入 success / ended**，不能沿用整段动作的最终结果（`robodojo/index.ts:330-331`）。
   - 服务端要在每个 policy 帧里带上逐步的判定结果，做法参照 RoboTwin；
   - 补一个测试：成功发生在 `go_home` 中途时，导出的 episode 要包含完整的回位过程。

**中**
1. **`follow_waypoints` 的夹爪默认值**：不传时沿用当前夹爪命令，或者把 `gripper` 改为必填（`primitives/waypoints.ts:79-97`）。
2. **Franka 的 waypoints**：每段按实测位姿计算，并报告是否真正到达（要求见 7.3）。
3. **result 记录 planner 类型**：
   - 记录 `planner: model|human|ensemble|fallback|finetuned|flash|replay`；
   - memory 和 explore 要能区分；
   - planner 导出默认排除人的回合，提供开关可以显式包含。
4. **所有机器人的 eval.sh**：`valid()` 要比对 `extras` 字段。并给 extras 写入 result 补测试。
5. **ManiSkill real2sim follower**：套用 RoboLab 的修复（b453bc9、bfe1644）：RELEASE 之后只竖直向上，并加上以 MV_UP 结束的规则。补一条抓取偏移不为零的测试。
6. **RoboDojo**：
   - `ground_truth_poses` 跳过 Dynamic 实例，兼容列表类型的 label；
   - 读取 `env.unstable_envs`，不稳定的 episode 标为 unstable，不计入评分；
   - eval.sh 按上游 SeedManager 的规则跳过不稳定布局、补足数量，并修正 `eval.sh:7-8` 的注释；
   - README 明确写出"跑在 Isaac Sim 6.1 上，成绩与官方榜单不可比"，并列出和上游的差异清单；
   - 保留一条走 Isaac Sim 5.1 原版的路径：补丁改成可选；
   - 有 GPU 的机器上用同一个策略和同一组布局在 5.1 和 6.1 上各跑一次做对比，没条件就写"未验证"；
   - `_apply_physx_settings` 遇到不存在的字段要报错；
   - DLAA 设置失败时要记录日志，不能静默吞掉。

**低**（按 `audit-b1a9e9d.md` 第三节顺手修；下面几条优先）：
- **硬件锁**：
  - 用 `O_NOFOLLOW` 打开锁文件，并检查锁目录的属主和权限；
  - 优先用序列号标识机械臂；
  - RLinf 和 Polymetis 对同一台机械臂使用相同的 id；
  - 相机设备也加锁；
  - 权限错误要翻译成"被占用"的提示。
- **网页工具**：锁定第三方包的版本；在文档里写明 prompt 注入的风险。
- **文档与实际挂载一致**：`align_wrist` 的说明、抓取建议器的适用范围、技能文档开头列出适用的机器人和需要的工具。
- **纯 units / code 模式**：不挂附加工具。
- **real2sim**：`merge_shards` 改成幂等；修正 RoboLab follow 的 `--randomize-xy`。
- **RoboDojo `env.step`**：失败结束时正确给出 terminated 或 truncated。

**交接里还没做的**（原有要求不变，这里只是列出，提醒不要漏掉）：
- XPolicyLab 客户端（`handoff-xpolicylab-robodojo.md` 第 1 节）；
- 第 6、7 阶段中未完成或偏离要求的各项，见 `audit-b1a9e9d.md` 第四节。其中 6.3 物体记忆要补上"按名字 / 图像检索参考资产"，或者在交接里说明和 OpenETA 的差异、由用户决定。

---

### 1.9 a2c880c 审计出的新问题（依据：`audit-a2c880c.md`，编号沿用该文件第二节）

**已完成，不用再做**：
- 1.7 的高危项：非 root 沙箱、`/gumi-replay`、dashboard `/primitive`、手动调用互斥、标定、CHANGELOG；
- 1.6 全部；
- 1.2 抓取链；
- 上一轮导出与 ensemble 的高危项；
- 3.1 LIBERO 原版提示词；
- 5 阶段的 XPolicyLab 客户端主体。

**高（先做，按顺序）**
1. **RoboDojo flywheel 逐步 success**：1.8 高 #1 仍未修（`robodojo/index.ts:371`）。
2. **被放弃的原语收不到停止**：放弃时置位并保持一个停止标志，让被放弃的线程从 `stop_requested()` 读到真（`code_exec.py:1387-1429`、`rpc_facade.py:194`）。
   - 补一个测试：原语先阻塞，放弃之后它剩下的动作必须停下。
3. **双臂 Franka 的 `--ik`**：
   - 后端不是 curobo 时，启动直接拒绝；
   - "robot obstacles need collision spheres" 按 `blocked` 处理，不能当作 `unknown`。
4. **ManiSkill DrawTriangle / DrawSVG**：
   - 按上游的 `max_episode_steps`（300 / 500）截断；
   - 在任务说明里补上"所有画下的点都必须在轮廓附近"这条成功条件（中 #29）。

**中**
- **感知与抓取**：
  - #1 Robosuite 和 LIBERO 的抓取规划器改为共享感知的 book（`masks=perception.book`），参照 Genesis；
  - #3 `follow_waypoints` 某段失败后停下，返回 `stopped: "stalled"`；
  - #4 GraspNet-1B / AnyGrasp 把 `depth` 计入 EEF 位置。有条件的话，用 GSNet 真正执行一次抓取；
  - #5 Metaworld / Genesis 按夹爪方向筛选候选；
  - #2 在文档里写明 Franka 只在路点处检查碰撞，或者改为执行关节轨迹；
  - #6 cuRobo 保留 link0/1 的碰撞球，并在有 GPU 的机器上确认。
- **代码模式**：
  - #7 panda_pair 的 `servo` 读取对应臂的 TCP；
  - #8 widowx250s 的 `_split` 裁剪到 [-1, 1]；
  - #9 RoboCasa 锁存中途的成功；
  - #10 RoboCasa 的进度类原语只放进 privileged 档，high 档默认要有运动原语；
  - #11 代码预算写进 `code.result()` 和 eval.sh 的配置键，oracle 运行时自动放宽上限；
  - #12 非 root 检查挪到启动或 preflight；远端 `URL#token` 不拒绝；文档写明真机的做法。
- **运行时**：
  - #13 `/robot-check` 解析 `#token=` 并携带 token（和 `attach()` 共用解析）；
  - #14 LIBERO `--env` 改用 `attach()`；
  - #15 `--approval` 按规格 2.2 改：`standard` 只确认高风险动作，真机默认 `human`；
  - #16 审批和原有确认合并成一次；
  - #17 `--serve-models` 只认自己子进程的服务：就绪检查核对 pid 或实例 id；退出时只关自己起的；
  - #18 `/gumi-replay` 在没有 dashboard 时也能停下，比如响应 Esc，或者要求开着 `--dashboard`。
- **XPolicyLab / VLA**：
  - #19 env_cfg 的 franka / piper 维度改回上游的双臂值，补上 `arx_x5`；README 不再让用户覆盖 XPolicyLab 的 env_cfg；dual Franka 和 dual Piper 挂载 xpolicy，并写明需要微调；
  - #20 RoboDojo 挂载 `xpolicy_act`，补权重下载脚本；
  - #21 OFT 的 `libero_all` 按套件选 unnorm key，或者拒绝与 key 不符的套件；`vla.info` 报告 key；
  - #22 LIBERO 步骤快照加开关，eval 默认关闭或事后清理，tmp 目录要删除。
- **GUMI / units**（HumanCLAW 第 2 阶段前必须修）：
  - #23 观察动作不能用终止单元：加一个非终止的 look，或者直接取观测；
  - #24 dashboard 的按键按标签匹配；
  - #25 记录实际执行的单元和参数；
  - #26 自定义词表的 `act` 在 video_ref 失败时拒绝；`letters_blind` 过滤结果里的方向信息；
  - #27 stage 上限按臂分开计数。
- **机器人**：
  - #28 bridge WidowX 的 fps 改为 5；
  - #30 1.8 中 #6 RoboDojo 剩余各项。
- **未经批准删除，补回来**（用户要求全部迁移）：
  - #31 `solve_ik` / `move_to_joints`（以及 2.9.1 中列出的 `traj_plan` / `move_along_trajectory`），在有关节控制的机器人上提供；
  - #32 Genesis 恢复上游的成功规则。若要保留新规则，改成可选，并在结果里记录用的是哪条规则；
  - #33 的几项在交接里逐条说明，由用户决定；README 不要引用仓库里没有的 specs。
- **#34 原语清单**：7068a96 不是 2.9.1b 要的静态共享清单，2.9.1b 照旧要做。做的时候：
  - 顺带修掉 `KNOWN` 白名单里的三处命名漂移；
  - 把 13 个 registry 的 `example` 迁进清单；
  - S3、S4 和 `raw` 档按 2.9 的新含义重新标注。

**低**：见 `audit-a2c880c.md` 第三节。优先做：
- Piper `step_pair` 之后让 id 失效；
- 给 Franka、双臂 Franka 和 LIBERO 补 Molmo `point`，或者改正提交信息里"每个机器人"的说法；
- 把 `suggest_grasp` 和 RoboDojo `locate` 加进 NON_MOTION；
- result 记录 planner 类型（1.8 中 #3，仍未做）；
- 上下文版本补 git dirty 标记和实际使用的模型 id；
- LIBERO 的 low 档补示例，否则 S3 等于 S4。

---

## 第 2 阶段：补迁移规格里明确要、但还没做的

### 2.1 `run_code` 推广到其他仿真（CaP-X，最重要）
- **现状**：只有 LIBERO 注册了 `code.run`（`libero/env_server.py:336-340` 的 `CodeRunner`），也只有 LIBERO 的 TS 声明了 `code:`（`libero/index.ts:316`）。
- **先做 Robosuite**（CaP-X 的核心基准），再做 BEHAVIOR、MetaWorld、Genesis、ManiSkill：
  1. 服务端照 LIBERO 的写法：`registry_primitives` + `CodeRunner` + `_begin_run` / `_finish_run`。计步和成功判定要放在 step / chunk_step 里，参考 515f708。
  2. 低档 `raw_obs` 必须过滤掉物体状态。MetaWorld 的 raw obs 带物体状态，要特别注意。
  3. TS 端声明 `code:` 规格，包括 observe、refuse、instruction。
- **S4 档**：`code/index.ts:43` 的 `TIERS` 加上 `low-noexamples`，Python 端已经支持。
- **CaP-X 的 17 个人类 oracle 程序**（cap-x `env_configs/human_oracle_code/*`）：
  - 放到 `packages/embodied/src/<robot>/oracle/`；
  - 加一个 `--code-oracle <file>` 参数，直接执行 oracle，不经过 LLM，作为人类基线；
  - 验收：Robosuite 的 oracle 用 `run_code` 能跑通。这一条只能在有 robosuite 的环境里做，没有的话写"未上机验证"，用 fake 测接口。

### 2.2 审批模式（OpenETA spec:109）
- **参考**：OpenETA 的 `agent/runtime/supervision.py`。
- **三种模式**：
  - `human`：每个会动机器人的工具都要人确认；
  - `standard`：只有高风险动作要确认；
  - `reviewed`：由一个审查 VLM 判断，拒绝时把理由返回给 agent。
- **pi 的写法**：用 `tool_call` hook 拦截动作，加一个 `--approval human|standard|reviewed` 参数。审查模型复用 `units/vlm.ts` 的 `askVlm`，并计入 `--max-cost`。
- **挂载**：做成共享模块，挂在所有机器人上。真机默认 `human`，仿真默认关闭。
- **注意**：和 operator、代码模式真机的确认要协调好，不能重复弹窗。

### 2.3 几何工具集（OpenETA spec:112，openeta-for-codex 分支）
- **参考**：openeta-for-codex 分支的 `tools/embodied_mcp_server.py`，包括：
  - `mark_point`：在点云正交视图上标点；
  - 按接近方向和夹爪方向执行的 `move_to`；
  - 闭合预览；
  - 残差距离和接触判断。
- **先读透这个分支，再在服务端做成原语**（进 `code.api`，工具和代码模式都能用）。
- **先挂 LIBERO**，这是 OpenETA 70.8% 成绩的来源；Franka 其次。

### 2.4 并行多模型推理 provider（CaP-X spec 5.1，CaP-Agent0 ensemble）
- **参考**：cap-x `llm/client.py:437-744`、`hillclimb/*.yaml`。
- **pi 的写法**：注册一个 provider，比如 `ensemble.ts`，参数叫 `--ensemble-*`。它把请求同时发给 N 个模型，再由一个综合模型合并结果。
- **要求**：
  - 费用计入 `--max-cost`；
  - 服从 `--max-api-concurrency`；
  - 和 fallback 的关系写清楚。

### 2.5 planner 会话导出为训练数据（CaP-X spec 5.2）
- **参考**：cap-x `cli/prepare_verl_dataset.py`。
- **做法**：在 `services/pi_embodied_services/flywheel/` 加 `planner_export.py`。把 pi 会话 JSONL（包括 system、user、工具调用、工具结果、图像）导出成 SFT 和 RL 可用的格式，并带上 episode 的成败作为奖励。
- **不做**：GRPO 训练本身放在仓库外。

### 2.6 上下文版本记录（OpenETA spec:111）
- **现状**：`robot.ts:548` 只记了 `code_api_digest`。
- **补充**：在 result.json 里加上：
  - SYSTEM.md 渲染后的 SHA-256；
  - memory 语料的摘要；
  - 仓库的 git commit 和 dirty 标记；
  - 模型 id 和 thinking 级别。

### 2.7 闭环规则（OpenETA）
- 把 OpenETA `agent/prompts/embodied_closed_loop.md` 的通用规则抽成一个共享段落，由 `robot.ts` 注入；各机器人的 SYSTEM.md 只保留自己特有的规则。
- 加上"运动未知时重新观测"的关卡：一次运动 RPC 超时后，下一次运动前必须先做一次观测。

### 2.8 预算
- 在 `robot.ts:223-233` 附近加 `--max-tool-calls` 和 `--max-tokens-total` 两个上限。

### 2.9 统一的档位参数：`--tier`（CaP-X 的 8 档）和 `--preset`（各源仓库的原生设定）
放在 2.1（`run_code` 推广）之后做。

**目标**：用户只写 `--tier M3` 或 `--preset showharness`，不用再换算成要开哪几个开关。

**依据**：审计 session 核实过以下来源：
- CaP-X 论文第 3 节和表 1：arxiv.org/html/2603.22435v2；
- `capx/envs/trial.py:225-300`：视觉反馈和 VDM 的逻辑；
- `capx/utils/launch_utils.py:135-137`：`use_visual_feedback` 默认是 False；
- `capx/integrations/franka/control.py`、`control_reduced.py`、`control_privileged.py`：三档 API；
- `env_configs/cube_stack/*.yaml`：8 档对应的配置。

#### 2.9.1 CaP-X 8 档的准确定义
| 档 | 感知 | 原语 | 示例 | 轮次 | 每轮的反馈 |
|---|---|---|---|---|---|
| S1 | 真值（mask、物体位姿） | 高层 | 有 | 单轮 | 无 |
| S2 | 真实感知，有噪声 | 高层 | 有 | 单轮 | 无 |
| S3 | 真实感知，有噪声 | 低层 | 有 | 单轮 | 无 |
| S4 | 真实感知，有噪声 | 低层 | **无** | 单轮 | 无 |
| M1 | 有噪声 | 高层 | 有 | 多轮 | 只有文字（stdout/stderr） |
| M2 | 有噪声 | 高层 | 有 | 多轮 | 文字 + 当前 RGB 图 |
| M3 | 有噪声 | 高层 | 有 | 多轮 | **只给 VDM 文字**：初始场景描述 + 每轮的前后差分，不给原图 |
| M4 | 有噪声 | 低层 | 有 | 多轮 | 只给 VDM 文字 |

**"高层"和"低层"具体指什么**（这是之前理解错的地方）：
- **高层**（`FrankaControlApi`）：任务级的语义函数，感知藏在函数内部：
  - `get_object_pose(name)`、`sample_grasp_pose`、`goto_pose`、`home_pose`、`open_gripper` / `close_gripper`。
- **S1**（`FrankaControlPrivilegedApi`）：接口和高层完全一样，只是 `get_object_pose` 直接返回仿真真值。
- **低层**（`FrankaControlApiReduced`）：拼积木用的原件，**包括感知**：
  - 感知：`get_observation`、SAM2 / SAM3（文字和点提示）、OWL-ViT、Molmo 指点、3D 包围盒、`plan_grasp`；
  - 运动：`solve_ik`、`move_to_joints`、`traj_plan`、`move_along_trajectory`；
  - 夹爪、双臂版本。
- **S 档的提示词里没有图像**。但低层 API 能在代码里调 `get_observation` 取图，所以 S3/S4 并不是"盲写"。

#### 2.9.1b 原语清单合并为一份静态共享清单（用户已定，前置）
**决定**：TS 和 Python 两份原语清单合并成**一份静态声明文件**，两边都从这份文件读取。
- 这和 9-26"不从服务端 `code.api` 运行时自动注册工具"的决定不冲突：清单是仓库里的静态文件，TS 在**扩展加载时**读取，不依赖 env server 已经启动。
- 工具快照测试、`--tools` 过滤、`/embodied-setup` 都照常工作。

**清单格式**：每个机器人一个文件，建议放在 `packages/embodied/primitives/<robot>.json`（位置可以调整，但 TS 包和 services 都要能读到；Python 端按仓库相对路径读取，并在 `services/README.md` 写明）。每条原语包含：
- `name`：工具名，也是代码模式里的函数名；
- `side`：`env`（由 env server 执行）、`ts`（只在 TS 端，比如 `pi0_pick`、`check_attached`、`web_search`、`act`、`finish`）或 `code`（只在代码模式里提供）；
- `method`：`side=env` 时对应的 RPC 方法；
- `params`：每个参数的名字、类型（number / string / boolean / array / vec3 / enum 值）、是否必填、单位、取值范围、说明；
- `tiers`：`high` / `low` / `raw` / `privileged`，按 2.9.2 的新定义；
- `mutating`：会不会动机器人；
- `requires`：可选，这条原语依赖的开关或能力（如 `ik`、`curobo`、`privileged`、`sam3`）；不满足时 TS 不注册，服务端也不放进白名单；
- `doc.tool`：工具模式下给模型看的说明；
- `doc.code`：代码模式的 docstring，包括 `Example:` 段，S4 档自动去掉示例；
- 机器人之间共享的原语，比如抓取、感知、`follow_waypoints`，写在 `primitives/common/*.json` 里，各机器人按名字引用，同时可以覆盖说明或范围，避免十几个文件重复。

**TS 端**：
- 新写一个加载器，读取清单，生成 TypeBox 参数 schema、工具名和说明，交给 `robot.tool` 注册。
- 工具的执行函数仍然在 TS 里写，但只能：
  - `side=env` 的：调用清单里声明的 `method`，再加上结果展示的逻辑（附图、状态、录制）；
  - `side=ts` 的：实现 TS 端的逻辑。
- 现有 `primitives/*.ts` 和各机器人 `index.ts` 里手写的 schema 和说明，全部迁进清单，删掉原来的手写部分。

**Python 端：`code.api` 瘦身（用户已定）**
- **删掉**：各 `robots/*/primitives.py` 里手写的声明，也就是名字、参数、档位、`example`，全部迁进清单。各机器人 facade 只保留执行方法本身。
- **保留白名单**：`CodeApi.resolve` 保留，改为启动时从清单构建。
  - 原因：代码模式的程序在服务端执行，TS 管不到。
  - 程序只能调用清单里 `side=env` 或 `side=code`、且属于本次档位的原语，参数也必须在清单里声明。
  - 其他 facade 方法（`stop`、`reset`、`code.set_limits` 等）一律拒绝。
- **`code.api` RPC 只返回两样东西，不再返回原语列表**（prompt 由 TS 直接读清单渲染）：
  1. `manifest_sha256`：服务端读到的清单摘要。TS 启动时和自己读到的比对，不一致就拒绝启动（fail closed），并提示两边版本不同。这主要针对 `--robot-env URL` 连到的远端服务。
  2. `available`：本次实际可用的原语名。
     - 由清单每条原语的 `requires` 字段（如 `["ik"]`、`["curobo"]`、`["privileged"]`）加上服务端的实际状态算出来。
     - TS 注册工具时用同一套 `requires` 规则过滤，并和 `available` 比对。只要有一条 TS 认为可用而服务端没有，就启动报错。
- **启动自检**：服务端启动时逐条对照清单，满足以下任一情况就报错退出：
  - `side=env` 的原语在 facade 上找不到 `method`；
  - facade 上的 RPC 方法既不在清单里，也没列为内部方法。
- **记录**：`code_tier_digest` 改为按"清单摘要 + 档位 + 可用集合"计算，写入 result.json。

**执行逻辑只保留一份**：`side=env` 的原语，工具模式和代码模式必须走同一个服务端方法。
- 例子：LIBERO 的 `move_to` 现在是在 TS 端循环调 `env.step`，要改成调服务端的 `env.move_to`。这样两种模式的限位、stop 处理、计步都是同一套。
- 其他有类似 TS 端逻辑的，按同样原则逐个改。

**检查（测试）**：
1. 清单的 schema 校验：字段齐全，每条都有 tier，`side=env` 必须写 `method`。
2. 每个机器人：
   - 清单里 `side=env` 的每条原语，服务端 facade 上都有对应的 RPC 方法（在 `services/tests` 里构造 facade 检查，不启动仿真）；
   - `side=ts` 的每条原语，TS 端都有实现；
   - TS 端注册的工具、代码模式能调用的函数，都来自清单，没有清单之外的。
3. 工具快照和 `code.api` digest 随清单生成，迁移那一次统一更新，之后以清单为准。
4. **反向检查**：服务端有 RPC 方法，但清单里没有、也没被列为"内部方法"的，报出来。Robosuite 的 `plan_grasp` 模型调不到，就是这一类。

**顺序**：先做这一节，再做 2.9.2 的档位重新标注。标注直接写进清单，只改一处。

**迁移范围**：全部机器人，包括 RoboDojo、UR5e、Piper，以及 OpenETA 附加功能里新增的 waypoints / align_wrist / suggest_grasp。可以一个机器人一次提交，每次提交都保持测试全绿。

#### 2.9.2 先修正我们原语注册表的档位标签（前置，必须先做）
**现状**：和 CaP-X 对不上。以 LIBERO 的 `robots/libero/primitives.py` 为例：
- `low` 只有 `step`、`chunk_step`、`raw_obs`、`move_delta`、`rotate_delta`，没有感知，比 CaP-X 的低层还低；
- `high` 是 `segment`、`back_project`、`move_to`、`execute_grasp`，更接近 CaP-X 的低层，外加一部分语义动作。

所以目前跑出的 S3/S4 结果和 CaP-X 不可比。

**要改成**：每个原语的 `tiers` 取以下四个值之一：
- `high`：语义函数。缺的要补，按 CaP-X 的签名：`get_object_pose(name)`（内部走 SAM3 + 深度反投影，或者 Molmo）、`sample_grasp_pose(name)`（内部走 `plan_grasp`）、`goto_pose(pos, quat)`、`home_pose()`、`open_gripper` / `close_gripper`。
  - 先在 LIBERO、Robosuite、Franka 上补齐，这三个都有感知和抓取后端；
  - 其余机器人不能提供的，在 `--tier` 选到高层时启动报错。
- `low`：感知、IK、关节 / 轨迹这些原件：`get_observation`、`segment`、`point`、`back_project`、`plan_grasp`、`preview_reach` / `solve_ik`、`move_to`、`move_delta`、`rotate_*`、`set_gripper`、`execute_*`。
- `raw`（**新增**）：只有底层 step 和原始观测：`step`、`chunk_step`、`raw_obs`。这就是现在的 low，保留下来，Show-Harness、HumanCLAW 这类"每步选一个动作、没有感知原语"的设定要用它。
- `privileged`：`get_object_pose` 的真值版本，以及 `ground_truth_poses`。

要求：
- 每个机器人的每个原语都必须有标签，没有标签的在测试里报出来；
- 改完之后更新 tool-schema 快照和 `code.api` 的 digest。

#### 2.9.3 展开规则
`--tier` 展开成四个正交的轴，每个轴也可以单独使用：

| 档 | `--turns` | `--feedback` | `--api` | `--privileged` |
|---|---|---|---|---|
| S1 | single | none | high（真值版 `get_object_pose`） | 开 |
| S2 | single | none | high | 关 |
| S3 | single | none | low | 关 |
| S4 | single | none | low-noexamples | 关 |
| M1 | multi | text | high | 关 |
| M2 | multi | image | high | 关 |
| M3 | multi | vdm | high | 关 |
| M4 | multi | vdm | low | 关 |

**`--turns single|multi`（新增）**
- `single`：只执行一次（代码模式下一次 `run_code`，工具模式下一轮工具调用），然后按环境状态自动 finish，不再把执行结果反馈给模型。
- **单轮时提示词里不放任何图像**，与 CaP-X 一致；程序仍然可以通过 `get_observation` 自己取图。

**`--feedback none|text|image|vdm|image+vdm`（新增）**
- 实现方式：用 context hook 从初始提示和所有工具结果里去掉图像，换成一句说明文字。**录制、视频、flywheel 仍然要拿到图像。**
- `vdm`：去掉图像，改成 VDM 文字：初始场景描述（CaP-X 的 `_describe_initial_scene`）+ 每轮差分。
- `image+vdm`：就是现在 `--vdm` 的行为，不属于 CaP-X 8 档；`--vdm` 保留为它的别名。

**`--api high|low|low-noexamples|raw`**
- 代码模式：`--code-api` 保留为别名。
- 工具模式：按 2.9.2 的标签过滤机器人的工具。

**冲突和不可用时的处理**
- `--tier` 和显式写的某个轴冲突时直接报错，不要静默覆盖。
- 机器人不支持的组合，启动时报错。例如：真机没有 S1；没有 `run_code` 的机器人不能跑代码模式的 S 档；没配 VDM 模型不能跑 M3/M4；没有高层语义函数的机器人不能跑 S1、S2、M1、M2、M3。
- `--privileged` 叠加到 M 档是允许的，但要记成 `M2+privileged` 这样的组合档，不算标准 8 档。

#### 2.9.4 `--preset`：各源仓库的原生设定（与 `--tier` 互斥）
其他仓库都不是写代码，而是"每步调工具或选动作"，只能类比到 CaP-X 的档位。它们的原生设定都是**多轮 + 图像反馈**，区别在原语的抽象层次。

| preset | 展开 | 说明 |
|---|---|---|
| `showharness` | multi / image / raw + units | 每步选一个 2 cm 单位动作，不给感知原语，比 CaP-X 的低层还低 |
| `humanclaw` | multi / image / raw + 自定义运动词表 + verifier | 同上。论文模式走 `humanclaw-psv` provider，见 `handoff-humanclaw.md` |
| `rpent` | multi / image / low + VLA 技能 + memory | 感知工具 + `move_to` + `pi0_pick`，长指南算作示例 |
| `openeta` | multi / image+vdm / low + 组合的抓取放置 | 默认是图像加 visual delta，不等于 CaP-X 的 M3 |
| `capx-<档>` | 等同于 `--tier <档>` | 写法统一 |

**记录**：
- `result.json` 写 `tier` 或 `preset`，以及展开后的 turns / feedback / api / privileged；
- eval.sh 和 eval-parallel 按这些字段分输出目录，并拒绝在同一个目录里混入不同设定的结果；
- README 加上档位表和 preset 表。

#### 2.9.5 测试
- 8 档和各个 preset 的展开结果；
- 冲突时报错、不支持时报错；
- `--feedback text` 和 `--feedback vdm` 下，模型上下文里没有图像，但录制里有；
- `single` 执行一次后自动结束，而且提示词里没有图像；
- 工具模式下 `--api low` 和 `--api raw` 过滤后的工具列表；
- 注册表里每个原语都有标签。

---

## 第 3 阶段：影响基准结果可比性的

### 3.1 LIBERO 长提示词恢复原版（RPent，用户已决定）

**审计 session 的判断**：压缩是实打实的问题，不只是篇幅差异。

**现状**：
- `libero/SYSTEM.md` 39 行，对应 RPent `robots/libero/prompts/evaluate.py` 607 行。
- `libero/explore.md` 38 行，对应 RPent `prompts/explore.py` 451 行。
- RPent 的 3 份 guide 完全没迁：
  - `strict_hybrid_guide.md`（603 行）
  - `pro_hybrid_guide.md`（506 行）
  - `env_calibration.md`（165 行）

**丢掉的内容**：
- evaluate prompt 里 80 个任务验证过的 `PERCEPTION_ALGORITHM`；
- `LOCALIZATION`（定位表 + 最终就绪检查）；
- 完整的 `RULES`；
- `KEY_HYPERPARAMETERS`；
- 强制读 memory 并在 `strategy_notes` 里记录读了哪些文件的工作流（`WORKFLOW_STEPS`）；
- `PROVEN_LEVERS`；
- 本地 memory 三层的使用说明（`LOCAL_MEMORY_PROFILE`、`STEP_READ_LOCAL_MEMORY`）。

**为什么影响大**：
- RPent 公布的 LIBERO 成绩是用这套提示词跑出来的；少了它，pi-embodied 的成绩和 RPent 没法比。
- HF 上的 memory 语料（RLinf/RPent-memory）也是按这套工作流写的，提示词不教 agent 怎么查，memory 的作用会打折。
- guide 在 RPent 里不是注入 prompt 的，而是 evaluate prompt 让 agent 用文件工具去读（`evaluate.py:413-418`）。pi-embodied 里 memory 的读权限只放行 `global/suite/task_only/results`（`memory/index.ts:18`），所以就算把 guide 放进仓库，agent 现在也读不到。

**不能原样照搬的地方**：原版引用了 pi-embodied 没有的东西：
- 分步历史：`view_env_state({"step": 0})`、`agentview_high.png` 这类文件；
- RPent 的工具名和参数。

所以"恢复原版"是"内容全部恢复，引用改成 pi 的写法"，不是逐字粘贴。

**做法**：
1. **先做第 4 阶段第 4 项（LIBERO 分步历史）**，挪到这里作为前置：
   - 每一步的产物落盘；
   - `view_env_state`、`segment`、`back_project` 支持 `step` 参数。

   原版提示词的定位流程依赖回看第 0 步和高分辨率图。
2. **guide 放进 `packages/embodied/src/libero/guides/`，原文不改**。
   - 在 memory 的读权限里加一个只读的 `guides/` 根，或者提供一个专门的 `read_guide` 工具。
   - 提示词里写明路径，要求 agent 每份读一次，和 RPent 一致。
3. **evaluate prompt 按原版的段落顺序逐段迁进 `libero/SYSTEM.md`**：
   - ROLE_AND_EVALUATION、PROVEN_LEVERS、RUNTIME、GOAL、RULES、LOCALIZATION、PERCEPTION_ALGORITHM、WORKFLOW_STEPS、ALLOWED PRIMITIVES、KEY_HYPERPARAMETERS、OUTPUT_DISCIPLINE，以及两种 memory profile（HF 和本地，分别对应现有的 `memory-hf.md`、`memory-local.md`）。
   - 工具名、参数、图像名换成 pi 的。
   - 依赖可选工具的句子用 `[tool:x]...[/tool:x]` 包起来。
   - `{{memory_dir}}`、`{{reference_tag}}` 等变量接到 `before_agent_start` 的渲染里。
   - 原版里 pi 已经由框架保证的内容（比如一次只执行一个工具、图像修剪），改成一句说明，不要删掉规则本身。
4. **explore prompt 同样按原版逐段恢复**，改 `libero/explore.md`，DISTIL 部分对照 `distil.md`。
5. **精简版保留成可选项**：加 `--libero-prompt rpent|compact` 参数，默认 `rpent`。result.json 里记录用的是哪一版，配合 2.6 的上下文版本记录。eval.sh 按这个参数区分输出目录。
6. **验收**：
   - 写一个测试，逐段检查原版每一节的关键规则句都出现在渲染后的 prompt 里。可以用 RPent 源码的段落标题和关键短语列一张对照表，放进 `test/fixtures/`。
   - 另写一份对照文档 `libero/PROMPT_PORT.md`，逐段列出"原文位置 → 迁入位置 → 做了哪些替换"。
7. **其他机器人（RoboTwin、RoboCasa、Franka、dual Franka）也检查一遍**：对照 RPent 各自的 `prompts/` 和 `guides/`，看有没有同样的压缩。有就同样处理，并在交接里写明结果。审读 agent 只核对了 LIBERO。

### 3.2 ManiSkill 任务表（OpenETA）
- 310dce4 已经加了选机器人型号。任务还是约 52 个里的 14 个（`maniskill/index.ts:54-67`）。
- 按 OpenETA 的 `sim/env_registry.py:1019` 补全，每个任务都要核对成功判定和 units 方向。

### 3.3 Show-Harness 双臂 fine-tuned 模式
- 迁移 DualMvTokenRunner 和 v4 双臂提示词（dual_once、dual_twice、dual_chain），加上 dual_mvtoken_roles。
- 目前 `finetuned/templates/` 只有两个单臂模板，`finetuned/index.ts` 也只有单臂 provider。
- 挂在 dual Piper 和 dual Franka 上。

### 3.4 Show-Harness 的 stage_control
- 给 `units/index.ts` 的 `plan` 补上三样：每个阶段的步数上限、推进/重试、出错时回滚到最近的抓取阶段。
- `act` 支持双臂同步一步（L/R 成对下发）。

---

## 第 4 阶段：工程便利性（可以穿插着做）
1. **模型服务自动拉起**（RPent `robots/runtime.py`）：TS 端按参数启动 VLA、SAM3、Molmo 服务，并等它们就绪；任何一个失败就全部停掉。
2. **训练脱离 Show-Harness 代码库**：把 `rollouts_to_alpaca`、`register_dataset`、LoRA 配置、`llamafactory_extensions` 迁进 `services/pi_embodied_services/finetuned/`，去掉 `train.sh` 的 `SH=` 依赖。
3. **Show-Harness 的 real2sim 数据生成**：ManiSkill Scheme D（record_demos、follow_tokenize）、make_dataset、merge_shards、check_dataset，以及 RoboLab 的生成器。
4. **RPent 的 LIBERO 分步历史**：已挪到 3.1 作为前置步骤，在那里做。落盘的 `segment_NN.json` 也要保证 flash-generate 的 segment 锚点能用。
5. **affordance**：补上"画点后自我确认"的循环（最多 2 轮）。
6. **subgoal / affordance 条件的 fine-tuned 数据和提示词**：迁移 `generate_subgoals.py`、`generate_affordance.py`。
7. **dashboard 补功能**：撤回消息、直接执行原语的表单、LLM 连通性检查路由、会话和视频下载。
8. **`/robot-check`**：真的发一次最小的补全请求，不要只探测 `/models`。
9. **硬件调试工具**：
   - dual Franka 的手动原语调用器；
   - Franka 和 Piper 的标定采集脚本（Z 地板、位姿）；
   - Piper 的 ROS 启动脚本；
   - GUMI rollout 重放。
10. **感知能力推广到更多机器人**：
    - SAM3 选择/排除、深度增强：目前只在 Franka；
    - 指点工具和 `ground_set`：目前只在 BEHAVIOR；
    - 把 MolmoPoint 的 `ground_set` 接到 LIBERO 的 Flash（主相机和腕部相机一次定位）；
    - UR5e 纯 RGB 相机接 UniDepth。
11. **cuRobo**：接上 `ik.plan` 的调用方，执行运动时加碰撞检查。
12. **训练导出覆盖新仿真和 UR5e**：补 flywheel 规则文件。
13. **GPU 端到端测试套件**，以及"新增机器人"和"新增原语"的开发者指南。
14. **抓取后端改名，并新增 GraspNet-1Billion 和 GSNet**
    - **先改名**：`components/graspnet_server.py` 改成 `contact_graspnet_server.py`，`--graspnet` 改成 `--contact-graspnet`。
      - 原因：它实际是 Contact-GraspNet。按 OpenETA 的约定，"GraspNet" 只用来指坐标系。
      - 连带要改：`franka/index.ts` 的 `graspArgs`、各机器人的参数、`services/setup.sh` 的目标、pyproject 的 extra、README、测试。
    - **再新增** `components/graspnet1b_server.py`，用 `--model baseline|gsnet` 二选一：
      - `baseline`：graspnet-baseline，权重有 `checkpoint-rs` 和 `checkpoint-kn` 两个；
      - `gsnet`：Graspness。
    - **输出格式**：原生就是 GraspNet 坐标系，可以复用 `anygrasp_candidates` 的转换。
    - **接入**：`utils/grasp.py` 的 `BACKENDS` 加上 `graspnet1b`；TS 加 `--graspnet1b`。
    - **安装**：要编译 pointnet2 和 knn 的 CUDA 算子，放进单独的 venv 和 setup.sh 目标。
    - **测试**：GPU 部分标"未上机验证"。
15. **按"能接的都接上"推广各模块**。依据审计 session 的对照：只差接线的 A 类先做，要先补服务端能力的 B 类后做。
    - **A 类（只差接线）**：
      - Robosuite 在 TS 端注册 `plan_grasp`、`plan_place`、`check_attached`。服务端已经装了规划器，但现在模型调不到，属于 bug，优先做。
      - Robosuite、MetaWorld、Genesis、BEHAVIOR、UR5e 接 memory、explore、`reset`。UR5e 靠 operator 复位场景。
      - Genesis 接 VDM。
      - 完整感知接到所有机器人：`Perception.install(facade)` 本身是通用的；Piper 和 UR5e 的 webcam 用 UniDepth 补深度。
      - Molmo 的 `point` 接到所有机器人。
      - `run_code` 接到所有机器人（和 2.1 合并做）。
      - Robosuite 的 IK 预检：服务端已有，只缺 TS 工具。
    - **B 类（先补服务端能力）**：
      - ManiSkill、RoboLab 打开深度渲染；Piper 接 RealSense 深度，或者用 UniDepth。
      - 其余机器人的抓取链。前提是 RGB-D、内外参、`grasp_to_eef` 标定都齐；只有 `move_delta` 的机器人，在 TS 端把目标位姿拆成多步 delta。
      - 各机器人的 IK 预检：PyRoKi 加对应的 URDF。
      - BEHAVIOR 的 units：先在服务端加"相对当前末端小步移动"的原语。
      - 真机的 flywheel：要改成在服务端录制，**做之前先和用户确认**。

---

## 第 5 阶段：接着做 `handoff-xpolicylab-robodojo.md`
这一阶段是 XPolicyLab 通用客户端、数据对齐和 RoboDojo 仿真，按那份文档做。Franka 系列的 VLA 种子也还没做。

**用户的要求**：VLA 凡是能走 XPolicyLab 的都走 XPolicyLab。审计 session 已经把 XPolicyLab d6332bf 的 46 个 policy 逐个核过，结论如下：

**能走 XPolicyLab 的**
- **RoboDojo（双 ARX X5）**：约 35 个 policy 有公开权重。它们都放在 HF 数据集 `RoboDojo-Benchmark/RoboDojo` 的 `ckpt/RoboDojo/<policy>/RoboDojo-sim-arx_x5-<joint|ee>-<seed>/` 下，apache-2.0 协议，不设门槛。
  - 大多数 README 没写这个下载位置，只有 Xiaomi_Robotics_1 写了。
  - 要写一个下载脚本，按 policy 和 action_type 拉取。
- **RoboTwin 2.0（aloha_agilex）**：以下几个有公开权重：
  - Evo_1：`MINT-SJTU/Evo1_RoboTwin2_clean` 和 `_datascale`；
  - OLA_SEM：ModelScope `Kosmos524/ola_sem`；
  - FastWAM：`robotwin_uncond_3cam_384`，大概率能加载，但没有文档，要先验证。

  按原交接第 1 节，先用 Evo_1 验证。

**两个限制，写客户端时要照着做**
- XPolicyLab 没有自带环境客户端，`run_sim_env_client.sh` 调的是 RoboDojo / RoboTwin 自己仓库里的脚本。所以 pi-embodied 的 XPolicyLab 客户端要自己按协议实现观测打包和动作块执行，可以参考 RoboTwin 的 `scripts/eval_policy_xpolicylab.py`。
- 它所有的 env_cfg 都是双臂，大多数 adapter 要 3 路相机（head + 左腕 + 右腕）。

**不能走的：保持现状**
- **LIBERO**：没有环境客户端，没有单臂 env_cfg。XPolicyLab 里 OpenVLA_OFT、GR00T_N17、Pi_05 这几个 adapter 都写死了双臂和 3 路相机，加载不了 LIBERO 权重。所以现有的 OpenVLA、OFT、GR00T、Pi0.5 服务**保留**，不要迁。
- **单臂 Franka、单臂 Piper、UR5e**：没有单臂 env_cfg，也没有权重。
- **RoboCasa、ManiSkill、RoboLab、Robosuite、MetaWorld、Genesis、BEHAVIOR**：没有环境客户端。上游虽然有 OpenWAM 的 LIBERO / RoboCasa 版、X-WAM 的 robocasa 版等权重，但没有 adapter 能服务它们。
- **dual Franka、dual Piper 真机**：XPolicyLab 有 `franka`（[7,7]）和 `piper` 的 env_cfg，但没有公开权重，开源版也不支持真机评测。要用就得用我们自己采的数据微调。
  - 客户端要支持这两种本体，将来有了权重可以直接接上；
  - 把"需要微调"写进 README。

---

## 第 6 阶段：用户追加要的（迁移规格原本没列）

### 6.1 人工充当视觉模型的操作台（OpenETA `tools/manual_vlm_proxy.py`）
- 注册一个 pi provider，比如 `manual-vlm`。
- 流程：
  1. 模型请求到来时，把 prompt 和图像推到 dashboard 的一个新页面；
  2. 人在网页上写回复（包括工具调用）；
  3. 回复作为模型输出返回给 agent。
- 用途：人类基线，以及调试提示词。
- 录制：这类会话在 result.json 里标 `planner: human`，flywheel 和 memory 能区分它。
- 复用：`dashboard/index.ts` 已有的 SSE 和 `/message` 机制。

### 6.2 网页搜索（OpenETA `agent/tools/web_access.py`）
- 两个工具：`web_search` 和 `web_fetch`，用 `--web` 开关，默认关。
- 共享模块，挂在所有机器人上。
- 搜索后端可配置（参考 OpenETA 的实现）。
- 结果要截断。
- 抓取网页时拒绝内网地址，因为 env server 和 VLA 服务都在本机端口上。

### 6.3 物体记忆库（OpenETA `agent/tools/object_memory.py`，`retrieve_asset_reference`）
- OpenETA 的版本依赖一个私有服务。先读代码，确认接口和数据格式，再做一个本地实现，数据放在 memory 语料目录下。
- 工具：按物体名或图像检索参考资产，比如形状、抓取经验、尺寸。
- 如果私有服务的接口没法在本地复现，就先做本地版，并在交接里说明和原版的差异。

### 6.4 Show-Harness 的实验性插件：coords、mcq、action_ablation
- 在 units 的插件机制下（`units/index.ts`，参照 `--units-*` 插件的写法）各做成一个开关：
  - `coords`：让模型输出坐标，而不是单位动作；
  - `mcq`：把动作选择改成多选题；
  - `action_ablation`：论文里的消融实验，按配置屏蔽部分单位动作。
- 每个都要对照 Show-Harness 的 `plugins/<name>` 和对应的提示词，行为要一致。
- 在 README 里标注"实验性"。

### 6.5 Viser 3D 视图（CaP-X `web/server.py`）
- 在 dashboard 里嵌一个 Viser iframe，显示：
  - 当前场景的点云（有深度的相机）；
  - 末端位姿；
  - 抓取候选和放置候选；
  - 规划路径（如果有）。
- Viser 服务由 TS 按 `--viser` 开关拉起（Python，放进 `services/` 的一个 extra），数据从 env server 取。
- 真机上也要能用：Franka 和 UR5e 都有深度相机或可以接 UniDepth。

## 第 7 阶段：其余剩下的，用户也要

### 7.1 腕部视角对准（OpenETA `grasp_geometry.py:988,1200`）
- 功能：给定目标，计算一个让腕部相机正对目标的末端位姿，并执行对准。
- 做成服务端原语（进 `code.api`）。先挂 LIBERO 和 Franka。

### 7.2 VLM 抓取建议器（OpenETA `grasp_pose_advisor.py`）
- 功能：让 VLM 根据图像给出抓取部位和方向的建议，并和 `plan_grasp` 的候选结合：
  - 用于排序；
  - 或者在没有抓取后端时代替它。
- VLM 调用复用 `askVlm`，费用计入预算。

### 7.3 多路点轨迹 `follow_eef_trajectory`（OpenETA `registry.py:1907`）
- 功能：一次调用执行一串路点。每段都要：
  - 检查单步上限和工作空间；
  - 在段与段之间检查 stop；
  - 返回每段的实际到达情况。
- 仿真和真机都要做，真机仍然走服务端的安全检查。

### 7.4 OpenETA 技能文档（`agent/skills/pick.md`、`place.md`、`push.md`、`pull.md`、`stack.md`）
- 这些文档和 OpenETA 的工具集绑定，要改写成 pi-embodied 的工具名。
- 作为 pi 的 skill 放进 `packages/embodied/skills/`，参照已有的 `embodied-quickstart`。
- 每份在开头写明适用的机器人和需要的工具。

### 7.5 单机硬件锁（OpenETA `real/mcp/observation_core.py:82`）
- 同一台机器上，同一个真机（或同一个相机设备）同时只能被一个 env server 占用。
- 用文件锁，放在 `/tmp` 或可配置目录，锁住硬件标识：机械臂序列号、相机设备路径。
- 启动时拿不到锁就报错，并写明被哪个进程占用。
- Franka、dual Franka、Piper、UR5e 都要加。

### 7.6 GPT 网页操作员（Show-Harness `web_operator*.txt`）
- 功能：让一个 VLM 通过 GUMI 遥操页面自动生成示范。
- 做法：
  - VLM 看遥操页面上的实时画面，输出 GUMI 语法的指令（`w*3 a g`、`L:.. R:..`）；
  - 指令经 GUMI 下发；
  - 录制时来源标 `gpt-operator`。
- 这和"pi agent 用 `act` 直接驱动"的区别：它严格走人类遥操的同一条通路，数据格式和人类示范完全一致。
- 迁移 Show-Harness 的 web_operator 提示词。

### 7.7 VDM 视频差分变体（CaP-X `trial.py:393-462`）
- 功能：前后两帧之外，把动作期间录的视频片段也交给 VDM，让它描述变化。
- 用 `--vdm-video` 开关，复用 video.ts 的 action clip。
- 只对支持视频输入的模型开放；不支持的模型要报错说明原因。

## 验收
- 每条修复都要有回归测试。
- 第 1 阶段做完后，全量跑一次 TS 和 Python 测试。
- 每完成一个阶段，更新 `packages/embodied/README.md`，让模块挂载表和实际情况一致。
- 没法在本环境验证的（GPU、仿真器、真机），在提交说明里写清楚。
