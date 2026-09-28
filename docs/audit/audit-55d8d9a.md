# 审计：main @ 55d8d9a（a2c880c 之后的 8 个代码提交）

**方式**：本人逐个审读 diff，只读，没有改仓库。

**测试**：
- `npm run check` 通过；
- TS：664 个，653 通过、11 跳过（跳过的是需要 GPU 的 e2e）；
- Python：765 通过、29 跳过。

**范围**：

| 提交 | 内容 | 对应交接条目 |
|---|---|---|
| ef525e1 | `src/` 按层重构 | — |
| 55d8d9a | 放弃原语后停机 | 1.9 高 #2 |
| 14da359 | 绘图场景的步数上限和成功条件 | 1.9 高 #4 |
| 77ddf4f、2ae3b6a | AnyPlace 放置 | — |
| 41098eb、488e2d3、106c6c6 | real2sim | 1.8 中 #5 和低项 |

不在范围内：4d7d4ed 是审计文档本身。

**总体结论**：
- 1.9 的高 #2、#4 已修，修法正确；
- 1.8 的 real2sim 三项已修；
- 重构没有发现断掉的路径；
- 问题都在中低级别。
- 1.9 的高 #1（RoboDojo flywheel）和 #3（双臂 Franka `--ik` 失败放行）**还没有提交**。

---

## 一、各提交结论

### 1. ef525e1 目录分层重构：通过
- **结构**：`src/` 分成 robots、primitives、modes、capabilities、planner、observation、infra、scripts 八层。`package.json` 的 `pi.extensions` 已改为 `src/infra/setup/index.ts`。
- **旧路径**：在 md/ts/py/sh/json/mjs/yaml 里搜不到残留的旧式路径引用，比如 `src/libero`、`embodied/src/franka`（`docs/audit` 和 CHANGELOG 除外）。
- **相对路径层数**：`robot.ts` 的 `SERVICES`、`scripts/eval-parallel.sh`、各 `robots/*/eval.sh` 和 `serve.sh` 指到 `services/` 的层数都核对过，正确。
- **行为变化（低）**：上下文版本的模板键改成了新路径，比如 `robots/libero/SYSTEM.md`。所以重构前后写出的 result.json，模板名对不上。
  - 如果 eval.sh 的配置键包含模板键，在同一个旧输出目录里续跑会被当成不同配置。
  - 建议在 CHANGELOG 或 eval 说明里提一句。
- **仓库外脚本**：提交信息已提醒，box 上的启动脚本要把 `-e packages/embodied/src/<robot>` 改成 `src/robots/<robot>`。

### 2. 55d8d9a 原语被放弃后停机：通过（1.9 高 #2 已修）
- **机制**：
  - 放弃原语时，`halt_motion` 让 `stop_requested()` 在调用内外都为真，同时 `request_stop()` 停下正在进行的动作；
  - 另起一个线程等被放弃的线程返回，之后再 stop 一次，再解除 halt；
  - UR5e 的 `_require_movable` 在发出任何指令前检查停止标志。
- **测试**：覆盖了"原语醒来后想动被拒、stop 被调用两次、halt 最后解除"，以及 UR5e 各运动方法在 halt 下都被拒。
- **残留（中低）：Piper 的夹爪路径先发指令、后检查停止**（`piper/controller.py` 的 `_gripper`）。
  - 运动流在每个路点前都有 `_check_stop`，但 `_gripper` 先调用 `_set_width`，之后 `_await_gripper` → `_wait` 才检查。
  - 例子：一个卡住的原语醒来后执行 `step(gripper="open")`，会真的把爪张开一次，然后才抛出 Stopped。如果此时爪里夹着东西，东西会掉。
  - 修法：`_gripper` 开头先 `_check_stop()`。更稳妥的做法是在 `step` 入口、`_ensure_synced()` 之前统一检查，`move_to_joints` 同理。
- **残留（低）：Franka / 双臂 Franka 依赖后端自己的停止处理，本提交没有加测试覆盖。** 从代码看，`request_stop` 递增 generation 后，被放弃调用的 `begin_op(gen)` 会失效，大概率是安全的。建议补一条和 UR5e 同样的"halt 下拒绝运动"测试。

### 3. 14da359 绘图场景：通过（1.9 高 #4 已修）
- `DOT_LIMIT = {DrawTriangle: 300, DrawSVG: 500}`。到达上限时，`_step` 直接返回 `truncated` 和 `step_limit`，不再调用 env。
  - DrawSVG 按 `dots_dist` 的宽度取 500，比 `MAX_DOTS` 的 1000 更保守，而且是实测得出的，合理。
- 任务说明补全了第二半成功条件（"不要在离轮廓超过 2.5 cm / 10 cm 的地方落点"），也写出了步数预算。
- **低**：到达上限时，返回的 `info` 沿用上一步的 `_last_info`，success 也就沿用上一步。按上游逻辑，到上限时 episode 本来就结束了，所以可以接受。

### 4. 77ddf4f、2ae3b6a AnyPlace 放置：方向正确，有两处偏离需要记录
- **77ddf4f**：
  - AnyPlace 的训练数据是 z 轴朝上的世界坐标系，这里原先喂给它的是 OpenCV 相机坐标系下的点云，这是一个真 bug。
  - 修复做法：服务端先把点云转到世界系，再用 `inv(T_cw)·T_w·T_cw` 把结果变换转回相机系。数学核对无误。
  - `validate_cam2world` 会校验刚体性。
  - 新增拒绝规则：接近方向偏离竖直超过 45°、落点不在区域上方、落点过高，都会拒绝并给出原因；候选全被拒时报错。
- **2ae3b6a**：只保留模型给出的绕世界 z 轴的转角和落点。放下的高度改为"区域顶面 + 抓取时 EEF 离支撑面的高度 + 1 cm"。
  - 已核对：绕 z 取最接近的转角，用的是 `atan2(R10−R01, R00+R11)`，正确。
- **偏离 OpenETA / AnyPlace（中低，需要写进文档）**：
  - 模型预测的倾斜被整个丢掉了（结果里只报 `model_tilt_deg`）。于是 AnyPlace 退化成"选落点 xy 和朝向"，插入、斜放这类任务做不了。
  - 提交信息说明了原因：多任务检查点是插瓶子的，对碗给出 35–60° 倾斜。
  - 建议加一个开关（比如 `--place-keep-tilt`），或者至少在 README 和工具说明里写明。
- **中低：放置高度的估计有系统偏差**：
  - 支撑面 `_support_z` 取物体**可见点**第 5 百分位的高度。从相机视角看，物体底部往往被遮挡，所以估计值偏高，离地距离被低估，物体会被往下压，1 cm 余量可能不够。
  - `eef_above_support_m` 用的是**规划的**抓取高度 `at[2]`，不是闭合后实测的 EEF 高度。
  - 在 box 上（碗放到盘子上）结果是对的（z 0.906 对 0.902）。换成高物体、侧视相机时没有验证。
  - 修法：用执行抓取后实测的 EEF 高度；并把 `settled_m` 和估计出的支撑高度写进结果，方便排查。
- **低**：
  - 拒绝判断用的是调整高度**之前**的落点。模型给的落点高于区域 25 cm 以上时会被拒，哪怕把高度调下来之后本可以执行。应当先调整高度，再判断。
  - `settled` 只在已执行抓取（held）的路径上生效。按未执行的 grasp_id 规划放置时，仍然用模型的落点高度，最多可以悬空 25 cm 就松手。

### 5. real2sim 三个提交：通过（1.8 中 #5 已修，1.8 低项已修两条）
- **41098eb**：
  - RELEASE 之后只沿竖直方向上升，纠正那一轮也只竖直；
  - 最后一帧不是 MV_UP 时判为失败，`reason: bad_ending`；
  - 和 RoboLab 的修法一致。
  - 美中不足：仍然没有抓取偏移不为零的测试（1.8 要求补）。
- **488e2d3**：
  - `merge_shards` 按（分片目录名，rollout 名）去重，编号从已有的最大值往后接，所以 `--move` 不会再嵌套。
  - **低**：去重键只取分片目录的 basename。不同父目录下同名的分片（如 `run1/shard0` 和 `run2/shard0`）会被当成已合并而静默跳过。建议用绝对路径，或者碰到这种情况直接报错。
- **106c6c6**：
  - 录制时把 `randomize_xy_m` 写进 track；follow 默认沿用这个值，显式传入不一致的值就报错。
  - **低**：旧 track 没有这个字段，会默认成 `RANDOMIZE_XY_M`。如果旧 track 是用 `--randomize-xy 0` 录的，布局就对不上，而且不会报错。

## 二、1.9 仍未提交的高 / 中高项
- #1 RoboDojo flywheel 逐步 success；
- #3 双臂 Franka 的 `--ik` 失败放行；
- 1.9 的中项和两项补回（`solve_ik` / `move_to_joints`、Genesis 成功规则）都还没有提交。
