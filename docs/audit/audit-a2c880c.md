# 审计：main @ a2c880c（b1a9e9d 之后的 135 个提交）

**方式**：
- 八路并行审读，只读，没有改仓库。
- 各组的探测脚本放在 `scratchpad/audit6/g1..g8/`。
- 下面标"已核对"的条目，我本人在代码里复查过；标"复现"的，有对应脚本复现。

**测试**：全部通过。
- `npm run check` 通过；
- TS：663 个，652 通过、11 跳过（跳过的是需要 GPU 的 e2e）；
- Python：757 通过、29 跳过。

**改动规模**：325 个文件，+45310 / −2105 行。

**总体结论**：
- 这一批大体按交接文档推进，质量不错。
- 已修掉：
  - 1.7 的三个高危项（非 root 沙箱、`/gumi-replay`、`/robot-check` 的方法名）；
  - 1.6 的全部内容（ManiSkill、UR5e）；
  - 上一轮导出和 ensemble 的高危项。
- 做完的新功能：
  - `run_code` 推广到全部仿真，CaP-X oracle 程序逐字节一致；
  - 感知工具推广到各机器人；
  - cuRobo 和 GraspNet-1B 接入；
  - LIBERO 长提示词恢复原版（与 RPent 逐字节一致）；
  - XPolicyLab 客户端；
  - 审批、闭环、上下文版本、预算；
  - Viser 视图和 VDM 视频模式；
  - units 自定义词表、stage control、双臂 fine-tuned 模式；
  - GUMI 的 VLM 操作员。
- **遗留**：
  - **RoboDojo flywheel 按整段动作打成功标签的问题仍未修**（已核对，`robodojo/index.ts:371`）；
  - 本轮新引入了 3 个中高问题；
  - 有两项被删掉，但用户没有批准过。

---

## 一、高 / 中高

1. **【仍未修】RoboDojo 的 flywheel 按整段动作打成功标签**（`robodojo/index.ts:371`，已核对）
   - 每个 policy 帧拿到的仍是整段动作最终的 `r.success` / `r.truncated`。
   - 后果和上一轮相同：导出的 episode 在 `go_home` 第一步就截断了。
   - 另外四个仿真（ManiSkill、Metaworld、Genesis、Robosuite）已经改成逐步判定，只剩 RoboDojo。

2. **被放弃的原语会继续运行，而且再也收不到停止**（`code_exec.py:1387-1429` + `rpc_facade.py:194-201`，已核对并复现）
   - 原因：
     - `stop_requested()` 只在"有 call 正在进行"时才可能为真；
     - `code.run` 放弃超时的原语、返回之后，`_active_generation` 被清成 None；
     - 被放弃的线程从此读到的永远是 False，无论之后收到超时停止还是 Interrupt。
   - 复现：原语先阻塞 1.5 s，然后跑 5 步、每步都检查停止标志。结果 5 次都读到 False，在放弃之后、甚至显式 `request_stop()` 之后，5 步全部执行完。
   - 真机影响：UR5e（`control.py:289`）和 Piper（`controller.py:658`）都靠这个标志停止。一个卡在阻塞读取上的多段动作原语，醒来后会继续驱动机械臂。
   - 修法：每次 run 设一个停止标志，放弃时置为真并保持；或者对被放弃的线程让 `stop_requested()` 恒为真。

3. **双臂 Franka 开 `--ik`、用默认 pyroki 后端时，所有动作都不经规划执行，也没有两臂之间的碰撞检查**（g3，脚本复现）
   - 触发过程：
     - 双臂 facade 总会发送一个 `robot` 障碍物；
     - `IkFacade._obstacles` 抛出 "robot obstacles need collision spheres"（`ik_server.py:1343`）；
     - `MotionPlanner.plan` / `check` 把这个错误当作 `unknown`（`motion.py:278,356`）；
     - `unknown` 只打一条警告，然后直接执行原始的 `move_delta`。
   - 这是真机上的失败放行（fail-open）。
   - 修法：
     - 双臂 Franka 只有在 `ik.robots` 报告后端是 curobo 时才允许 `--ik`；
     - 或者把这个错误当作 `blocked`。

4. **DrawTriangle 跑满 300 步、DrawSVG 跑满 1000 步后，env 抛 IndexError**（ManiSkill `env_server.py:681,756`，g8 按源码核对）
   - 原因：
     - 每个控制步都会写入 `dots[draw_step]`；
     - 上游靠注册的 `max_episode_steps` 让 episode 提前结束，这里却传了 100000。
   - 大约 40–150 次决策就会碰到这个上限。错误表现为 RPC 报错，而不是一次失败的 episode。

## 二、中

**感知与抓取**
1. **Robosuite 和 LIBERO：`detect` 的 id 和抓取规划器的 id 来自两个计数器**（`robosuite/env_server.py:256`、`libero/env_server.py:343`，复现）
   - 抓取规划器在 facade 里先建好，没有传 `masks=`；之后 `install_perception` 又新建了一个 Epoch。
   - 两边都会发出 `d1`，于是 `plan_grasp(mask_id="d1")` 可能去抓另一个物体。
   - Genesis 和 Metaworld 的写法是对的，可以照着改。
2. **Franka 真机上的"无碰撞规划"只在路点处检查**
   - cuRobo 给出 48 个采样点，只保留 ≤10 个路点。
   - 每段先用笛卡尔 `move_delta` 平移，再用 `rotate_delta` 转动。关节路径可能与规划不同，也可能切角。
   - 至少要在文档里说明；更好的做法是执行关节轨迹。
3. **`follow_waypoints` 某段失败后仍继续走剩下的路点**（`motion.py:395-440`，复现）
   - 应当停下，返回 `stopped: "stalled"`。
4. **GraspNet-1B / AnyGrasp 的 `depth` 没用上**
   - 按 graspnetAPI 的约定，指尖在 translation + depth·x 处；这里把 EEF 放在 translation。
   - 结果抓取会浅 1–4 cm。GSNet 只验证了能出候选，从没真正执行过抓取。
5. **Metaworld / Genesis 的 `execute_grasp` 忽略候选的 yaw**
   - 应当按夹爪方向（mod π）筛选或排序候选。
6. **cuRobo 模型里整个删掉了 `panda_link0/1`**（疑似，需要 cuRobo 才能确认）
   - 双臂互检时漏掉了对方的基座柱。

**代码模式**
7. **ManiSkill 双臂（panda_pair）的 `servo` 必定失败**（g4、g8 都复现）
   - `_code_move_m` 读取 `state()["tcp_pos"]`，双臂的状态没有这个键，报 KeyError。
8. **ManiSkill widowx250s 的 `step` / `chunk_step` 没有限幅，移动距离也少算**（复现）
   - 这个控制器 `normalize_action=False`，`step([5,0,0,1])` 一步会走 0.5 m，但距离上限只按 0.1 m 计。
   - 修法：在 `_split` 里裁剪到 [-1, 1]。
9. **RoboCasa 不锁存中途的成功**
   - 只在程序结束时检查一次 `check_success()`。程序中途成功、之后又把物体碰走，就会记成失败。
10. **RoboCasa 的非特权程序能拿到物体状态**
    - `get_task_progress` 等原语用的是默认档位；`get_task_progress` 会返回物体距离。
    - high 档没有任何运动原语，而 `--code-api` 默认是 high。
11. **代码模式的预算没有记录，oracle 在默认预算下跑不完**
    - 10 个 eval.sh 都不记录 `--code-timeout/max-calls/max-move/helpers`。
    - oracle 用默认的 50 次调用上限：`two_arm_lift` 要 52 次，`wipe` 大约要 83 次、5 m。
    - 修法：oracle 运行时自动放宽上限，eval.sh 把这几项纳入配置键。
12. **1.7 的非 root 拒绝让真机上的代码模式基本用不了**
    - 拒绝发生在每次 `code.run` 里面。那时机械臂已经复位、操作员也已经确认过程序。
    - 真机通常以非 root 运行，所以每个程序都会被拒。
    - 修法：
      - 把这个检查挪到启动或 preflight；
      - 用 `--robot-env URL#token` 连接远端服务时，不在同一台机器上，不应拒绝；
      - 文档写明真机应以 root 启动服务端，或者配置专用的 uid。

**运行时机制**
13. **`/robot-check` 在所有要求 token 的服务上仍然失败**（`check.ts:366`，已核对）
    - 这包括全部仿真，以及开了 `--code` 的真机。
    - 请求从不带 token；地址写成 `URL#token=` 时，`/call` 会被拼进 URL 片段，得到 404。
14. **LIBERO 的 `--env` 连不上自己的服务**（`libero/index.ts:1854`，本轮之前就存在）
    - 这里用 `new RpcClient` 而不是 `attach()`，所以不解析 `#token=`。
15. **`--approval` 的语义和默认值与规格 2.2 不符**
    - `standard` 完全不设闸门，规格要求它"只确认高风险动作"；
    - 真机的默认值应该是 `human`，实际仍是 `standard`；
    - 这两处偏差都没有写进交接。
16. **`--approval human` 下，`run_code`、场景重置等操作要确认两次**
    - 审批闸门和原有的确认对话框各弹一次，彼此没有协调。
17. **`--serve-models` 可能挂到别的进程的服务上，并在退出时把它关掉**（阅读推断）
    - 两个 pi 使用同一端口时，SAM3 先加载模型、后绑定端口，这段时间里存在竞态。
    - 后果：B 以为服务就绪，其实挂到了 A 的服务上；B 退出时发送 shutdown，把 A 正在用的服务停掉。
18. **`/gumi-replay` 在没开 `--dashboard` 时无法中途停止**（阅读推断）
    - 智能体空闲时，命令拿到的 `signal` 是 undefined，按 Esc 不起作用。

**LIBERO、XPolicyLab、VLA**
19. **pi 的 XPolicyLab env_cfg 把 piper 和 franka 写成了单臂维度**（`xpolicy_env_cfg/robot/_robot_info.json`）
    - 上游是 [6,6] 和 [7,7]，所有 env_cfg 都是双臂。
    - README 让用户把 `env_cfg` 软链到 XPolicyLab 读取的位置，这会连带改掉策略服务端看到的维度，而且那里也没有 `arx_x5`。
    - 与规格的出入：规格说单臂不能走 XPolicyLab、应改为支持双臂。现在 Piper 只在单臂时挂载，dual_franka 没有挂载。
20. **RoboDojo 的 `xpolicy_act`（arx_x5）没有挂载**
    - `robodojo/index.ts:574` 的 TODO 还在，也没有权重下载脚本。
21. **OFT 的 `libero_all` 检查点**
    - `--unnorm-key` 在启动时固定，而 `suiteMismatch` 放行四个套件中的任意一个。
    - 例子：用 spatial 的归一化统计量去跑 goal 套件，不会报错，数值却是错的。
    - `vla.info` 也不报告当前用的是哪个 key。
22. **LIBERO 每步都存历史快照，占用磁盘（估算）**
    - 大约每步 4–8 MB，一个 40×50 的评测就是几十 GB。
    - 没有清理，也没有开关关掉。没传 `--output-dir` 时写到 tmp，也从不删除。

**GUMI、units**
23. **GUMI 把 `STOP` 当作"观察"动作**（`gumi/index.ts:916,980`、`operator.ts:313`）
    - 在自定义词表里 `STOP` 是终止单元，第一次观察就会结束 episode；HumanCLAW 正是这种情况。
    - 词表里没有 `STOP` 时，又拿不到第一帧观测。
    - 测试甚至断言了这个行为。
24. **自定义按键在 dashboard 里失效**（`page.html:581,486`）
    - 按键标签是 `"WALK(normal)"`，词表里是 `"WALK"`，匹配不上，按键被丢弃，按钮上也不显示按键提示。
25. （中低）**GUMI 记录的是用户输入的标签，不是实际执行的单元**
    - 例子：输入 `TURN(500)`，实际被裁到 120 执行，记录里仍是 500。
26. （中低）**自定义词表的 `act` 在 video_ref 失败时没有拒绝**；**`letters_blind` 模式下，Piper 的结果把被遮盖的方向泄露给了模型**。
27. （中低）**双臂的 stage 上限是全局共用的**；上游是每只臂各自计数。

**机器人与 flywheel**
28. （中低）**ManiSkill bridge WidowX 数据集的 fps 写成了 20，实际是 5 Hz**（`flywheel.py:73`），数据回放会快 4 倍。
29. **DrawTriangle / DrawSVG 的任务说明漏了一半成功条件**：所有画下的点都必须在轮廓附近，只要落笔偏出一次就再也无法成功。
30. **RoboDojo 上一轮遗留的问题仍未修**：
    - 不读 `unstable_envs`；
    - eval.sh 不跳过、也不补足不稳定布局；
    - 没有"与官方榜单不可比"的说明；
    - 没有 Isaac Sim 5.1 路径；
    - DLAA 的空 except 和 `setattr` 都没改。
    - 另外：本轮新增的任务表说明 48/54 个任务可跑，6 个在启动前就拒绝。

**未经批准删除（7ddf945 "Deliberately dropped"）**
31. **CaP-X 的关节空间原语 `solve_ik` / `move_to_joints`**
    - 这与交接 2.9.1/2.9.2 把它们列在 low 档相矛盾。
    - 删除理由写的是"Robosuite 用 OSC_POSE"，但这对 LIBERO、Franka 等不成立。
    - 用户说过全部都要，应当补回来。
32. **Genesis 的成功规则被改成"抬起 8 cm 并保持 5 步"**，影响与上游的可比性，也没有经过批准。
33. 较小的几项，同样没有批准：
    - 普通观测里去掉了真值；
    - Show-Harness `prompt_v5` 没有拷入（数据集是 gated 的）；
    - CaP-X 的 LLM 代理和 FastAPI/msgpack 服务被替换。
    - README 引用的"migration specs"不在仓库里。

**原语清单（对照 2.9.1b 做法 3）**
34. 7068a96 只做了一致性测试：同名的原语，其参数名是工具参数名的子集。静态共享清单根本没建（`packages/embodied/primitives/` 不存在）。
    - 还有三处命名漂移被写进白名单 `KNOWN`，而不是修掉：`franka.segment`、`robosuite.move_to`、`robosuite.move_delta`。
    - 同时，13 个手写 registry 又新增了 `example` 字段；S4 被硬编码成"low 去掉 examples"；eval.sh 按旧的 low 含义记录档位。之后改档位时，这些都要一并迁移。

## 三、低（摘要）
- 代码模式：
  - LIBERO 只有 0/17 个原语带示例，所以 S3 和 S4 完全相同；其他机器人的示例也不全；
  - 主线程服务上卡住的原语会让整次 run 的结果丢失；
  - BEHAVIOR 的回复带着 `q_score` / goal 进度。
- `--vdm-video`：
  - 没有检查模型是否支持视频；
  - 读取的是 FRAME_EVENT，不是动作片段；
  - 没有视频时静默退回图像差分。
- 上下文版本：
  - 缺 git dirty 标记和实际使用的模型 id；
  - memory 记录的是读过的文件，而不是整个语料的摘要；
  - Piper 单臂运行也记录了 `SYSTEM_DUAL.md` 的摘要。
- 预算与审批：
  - ensemble 下 `--max-tokens` 少算候选调用的 token；
  - `suggest_grasp` 和 RoboDojo 的 `locate` 被当成了运动工具。
- 导出：
  - 人工 planner 的运行默认被导出；
  - 1.8 中 #3 的 `planner` 字段仍然没有。
- 感知：
  - Piper 的 `step_pair` 之后 id 不失效；
  - Franka、双臂 Franka 和 LIBERO 主智能体没有 Molmo 的 `point` 工具（提交信息却写着"每个机器人"）；
  - PROTOCOL.md 仍写 ManiSkill/RoboLab 没有深度；
  - RoboDojo 腕部相机的深度没有用上。
- 其他：
  - Show-Harness 的 subgoal/affordance 模板在上游不存在，是重建的，与上游 LoRA 不匹配；
  - coords 在上游是 no-op，这里真的生效了；
  - operator prompt 删掉了两条规则。
- g1 里的其余几条：
  - Viser 的 token 从 environ 里 pop 之后，`/proc` 里仍然可见；
  - 标定文件只保留一份 `.bak`；
  - `ros_launch.sh arms --enable` 不带 mode 时参数错位。
- XPolicyLab：
  - Piper 丢弃了 roll 和 pitch，只用 yaw；
  - 夹爪开合用启发式映射；
  - 没有和真实服务端的协议级测试。

## 附录：纯网络安全（不排序）
- `--viser` 默认监听 0.0.0.0，无鉴权。Franka 上等于把真机相机画面和点云公开出去。
- dashboard 在 loopback 上仍然没有 token。`run_code` 程序本身有网络，能调用 `/message`、`/interrupt`、`/human/reply`。
- 真机上的 `run_code` 程序可以直接打开 socket：UR5e 的 30002/30003 端口没有认证，Piper 可以开 CAN raw socket，这样就绕过了所有服务端检查。建议真机的沙箱用户断网（netns 或 seccomp）。
- env 服务的 HTTP RPC 不检查 Host、Origin、Content-Type。没带 `--code` 的真机服务不要求 token，网页可以用 CSRF 驱动机械臂。
- IK RPC 没有 token，路点和障碍物的数量也不设上限。

## 已修复并核实的上轮问题
- **1.7 的高危项**：
  - 非 root 沙箱：复现确认已拒绝；
  - `/gumi-replay`：执行前确认、出错即停、有间隔、可以被 interrupt 打断；
  - dashboard 的 `/primitive`；
  - 手动调用与智能体的调用互斥；
  - dual Franka 服务端的越界检查（残留：服务端没有执行 pi 的 `--workspace-xy` / `--z-floor`，`chunk_step` 没有检查）；
  - 标定 `--write`；
  - CHANGELOG 和文档；
  - 子进程的 stderr。
- **1.6**：
  - ManiSkill 的 5 项全部修复；
  - UR5e 的 A–E 在 mock 下成立，按 ur_rtde 1.6.5 源码核对过，但未上真机。
- **上轮导出与 ensemble**：
  - 图片数量和占位符对得上；
  - VeRL 的 prompt 与图片对应；
  - reward 不再存结果；
  - 特权、操作员、explore 的过滤；
  - steering 不并入观测；
  - temperature 和逐候选超时。
- **1.2 的抓取链**：五项全部修复。
- **LIBERO 原版提示词**：三份指南与 RPent eecf206 逐字节一致，SYSTEM.md 逐节对应 evaluate.py。
- **XPolicyLab**：协议复用了上游 `WsModelClient`；观测格式、部署循环、整段动作的预先校验都正确。
- **CaP-X 的 17 个 oracle 程序**：逐字节一致。唯一一处替换（handover 改用非特权版本）有正当理由。
- **run_code 的原语和真值**：
  - 真值只在 privileged 下给出；
  - Metaworld、Robosuite、ManiSkill 没有状态泄露；
  - 真机代码模式需要 `--code-real`、`--operator` 和逐个程序确认。
- **RPC token**：
  - 不会出现在日志、argv 里，也到不了子进程；
  - root 下沙箱 uid 无法读取 pi 或服务端。
