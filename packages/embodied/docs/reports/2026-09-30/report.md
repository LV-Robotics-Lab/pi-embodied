# pi-embodied 版本统一与 Qwen 回归记录

日期：2026-09-30。版本统一、构建和测试已完成；任务能力尚未达到验收要求。HumanCLAW 导航已跑通，但真实坐下交互尚未通过。本记录是小规模固定任务回归，不代表完整 benchmark 成功率，也没有与 Muse 做等条件对照。

## 代码与部署

- 主仓库：`/root/autodl-tmp/pi`，GitHub `LV-Robotics-Lab/pi-embodied` 的 `main`。
- 上游：`earendil-works/pi`，本次同步到 `1b347794e2a630e4359f2584f4eea388145d0ddf`（pi 0.99.1）；合并提交 `e651dd7082392f2aef5d23b4d281164a85ac6bad`。
- 修复提交 `5486a0a2af5f869a175d64dd2d3f4f279dd09031`：HumanCLAW 在服务明确返回 `response_format_not_supported` 时，使用相同图像和提示重试普通文本 JSON；会话内缓存该能力。修复 Python 服务启动失败时两个 undefined PID 被误判为健康的问题，并增加回归测试。
- pi 模式坐下指导修订：`bf195630e8252961ee3d771ef33a3d22a4043642`、`2acfcaab08a77dedb42282702b90146bc9b0f569`。说明身体朝向、连续坐下阶段、目标座面和邻近家具区别。未改论文模式提示或成功判据。实测尚不能证明交互能力提升。
- 统一入口：`/usr/local/bin/pi` → 主仓库构建产物；版本输出 0.99.1。五个具身环境脚本及 `/root/autodl-tmp/pi-humanclaw/hc-run.sh` 已统一到该入口和主仓库 services。
- 原工作区 lockfile、差异和部署脚本备份：`/root/autodl-tmp/pi-deploy-backup-20260930/`。历史实验工作区保留。
- GPU 0：ninfer `d44ab58408aa389728cd8b1ee50179527e1f3e0d`，Qwen3.8-27B NVFP4，模型 ID `qwen3.8-27b`，pi 选择 `selfhost/qwen3.8-27b`。端点 `http://127.0.0.1:8000/v1`；健康检查通过。
- 权重：`neroued/Qwen3.8-27B-nvfp4-NInfer`，revision `a107ba1b5b0609d9ca90d5ed61439f5f5cc7d64d`，SHA256 `74d2c57145e6ff11d1d2faa79594477f9bc903a611af1fb20218189fbbb77d82`。
- 固定回归显式选择 Qwen；用户全局默认模型未修改。GPU 1 用于仿真。

## 工程验证

- 主仓库安装、离线构建、完整 `npm run check` 通过；最终检查日志 `/tmp/pi-final-check.log`。
- embodied Node 测试：735 通过，9 跳过，0 失败（744 总计）。日志 `/tmp/pi-unified-embodied-fixed.log`。
- Python services：925 通过，35 跳过，0 失败；ruff 通过。日志 `/tmp/pi-unified-services-mirror.log`。有 pytest timeout 配置插件缺失及数值 fixture 警告；mock 后端测试不等于真实任务性能。

## 固定 Qwen 任务回归

运行脚本 `/root/autodl-tmp/tools/qwen-regression-20260930.sh`。三种后端各一个固定任务、种子 0/1；thinking=low、units=true、最多 40 轮、每任务 300 秒，不加载 skills 或 prompt templates。没有降低判据或增加预算来改写失败结果。

任务在修复期间分批执行，下表列出结果文件记录的提交；它是记录时版本，不保证等于进程启动时版本。后续 HumanCLAW 提示修改未改变这三种机器人的任务控制代码。因此这是一轮集成回归，不是冻结单提交的模型比较实验。

合计成功 1/6，invalid 0。LIBERO 为 1/2，MetaWorld 和 ManiSkill 均为 0/2。LIBERO 保留了既有具身记忆配置，结果中的 memory_files 记录加载文件及散列；pi0、SAM3 服务未运行，该轮使用动作单元。这个结果不是清空记忆后的独立泛化评测。

| 用例 | 结果 | 环境步数 | 预算耗尽 | 记录提交 |
|---|---|---:|---|---|
| libero_spatial_t0_s0 | failure | 254 | time | 2acfcaab0 |
| libero_spatial_t0_s1 | success | 339 | 无 | 2acfcaab0 |
| PickCube-v1_s0 | failure | 259 | time | 5486a0a2a |
| PickCube-v1_s1 | failure | 277 | turns | 5486a0a2a |
| reach-v3_s0 | failure | 147 | 无 | e651dd708 |
| reach-v3_s1 | failure | 353 | time | e651dd708 |

## HumanCLAW 专项

HumanCLAW 原仓库版本 `c4f9351`。本次测试两个场景任务：`104348028_171512877_ep11_chair`（导航）和 `104348028_171512877_ep1_couch`（导航加坐下）。多次重试同一场景属于诊断，不是独立测试集。

ninfer 原先拒绝强制 JSON 输出参数，导致论文模式连续失败并回退到走路；已修复协商逻辑，真实任务能继续。原始协议失败尝试保存在 `qwen-paper-baseline-20260930`，中途停止，不计有效 benchmark。

注意：当前 result.json 的 success/status 表示 NavSR@20cm，坐下任务必须另外检查 InteractSR。InteractSR 要求最后一次坐下末帧中骨盆接触目标网格，并主动 STOP；靠近目标或模型口头声称成功都不够。

| 运行目录 | 步数 | NavSR@20cm | InteractSR | 距离（米） | 备注 |
|---|---:|---|---|---:|---|
| qwen-nav-formatfix-20260930 | 97 | None | None | None | 超时；末态指标未产出 |
| qwen-paper-formatfix-20260930 | 36 | True | False | 0.1111288070678711 | 坐下交互未通过 |
| qwen-pi-nav-20260930 | 46 | True | False | 0 | 导航任务，交互不适用 |
| qwen-pi-sit-20260930 | 18 | True | False | 0.06597909233493127 | 坐下交互未通过 |
| qwen-pi-sit-targetfix-20260930 | 74 | True | False | 0 | 坐下交互未通过 |

最终沙发复测 74 步、10 次 SIT、目标距离 0，NavSR 通过但 InteractSR=False。碰撞步比例 86.3%。官方渲染末帧显示人在家具间下蹲，未坐到目标沙发座面。此前 18 步试验停在沙发旁圆边桌处，也未实际坐下。仅调整提示不足以修复目标座面定位、无进展恢复及真实身体接触控制。

下一阶段应固定新评测提交和独立场景集，针对座面定位、动作执行后位移/接触反馈及反复碰撞的恢复机制逐项验证；任何新增反馈都需区分标准视觉设置与特权设置。此次未修改 benchmark 阈值或用导航成功替代交互成功。

## 证据位置

- 六任务原始输出：`/root/autodl-tmp/runs/qwen-regression-20260930/`。
- HumanCLAW 原始日志、逐步图像和轨迹：`/root/autodl-tmp/runs/humanclaw/` 下的上述同名目录。
- 最终官方回放：`/root/autodl-tmp/runs/humanclaw/qwen-pi-sit-targetfix-render-20260930/{ego,exo}.mp4`。
- 本报告旁的 `results.json` 收集全部最终 result.json，并保留各自远程绝对路径。
