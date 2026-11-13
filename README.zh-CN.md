# RoboCasa 原子能力双循环框架

本项目实现一个从空技能库开始、由模型逐步填写 `skill(api, params)` 的研究框架。
固定 P01–P40 定义能力槽位；任务探索产生可执行代码，独立维护循环负责参数化、重新验证和发布。
本仓库只发布框架源码、测试和示例配置；不包含已学习的 code tool、技能库、模型响应、仿真轨迹或实验产物。

## 两个循环如何衔接

```text
任务 instruction + 初始 RGB-D / 机器人测量 + 已发布技能
                 ↓
        规划完整的有序原子步骤及各步后验条件
                 ↓
探索 loop：尝试 active 版本 / 填写空函数 → 真实 env.step → host 验证
                 ↑                              ↓
           失败反馈 + 当前观测 ← 失败          成功经验 pending
                 │                              ↓
         必要时重新规划未完成后缀       维护 loop：参数化 distill
                                                ↓
                           新建环境 → 回放前缀 → 验证候选原子
                                                ↓
                            当前原子的全部已登记旧案例回归
                                                ↓
                               全通过 → 新版本 → active
```

单个 episode 内的代码尝试共享物理状态；后续代码接收失败后的观测，不会在每次 repair 前重置场景。
开启跨 episode 重试后，host 可在失败 episode 结束后显式新建环境；这一 reset 与同 episode 内的 repair 是两层操作。
预算由原生任务 horizon、每原子尝试次数、重规划次数和代码执行超时共同限制。
任务完成始终以原生 RoboCasa verdict 为准，函数返回值只承载诊断信息。
默认 `maintenance_trigger: "atomic"`：每个真正执行且通过 host 检查的原子步骤立即写入 pending，随后同步进入独立维护循环；无需等待整个 task 或 episode 成功。
维护时暂停当前探索环境，在另一 fresh 环境中验证；发布后继续原探索，后续 repair/replan 可立即看到并调用新能力。
后续任务失败、horizon 耗尽、reset 或 provider 出错都不撤销已发布的局部能力。维护异常单独记录，探索与 reset 继续进行。
`maintenance_trigger: "episode"` 可改为每轮结束维护，包括失败轮；`maintain_after_task: false` 或 `explore` 命令关闭自动维护。
也可以分别执行 `explore` 和 `maintain`。跳过的已满足步骤、未知验证结果、仅函数自报成功的代码均不能成为新能力证据。

维护在新建环境中按原 task/seed 重建起点，执行记录的前缀，再运行参数化候选。
新经验保存 `entry_observation`、`entry_trace`（之前所有真实执行，包括失败尝试的物理准备）和当前原子的完整 `attempt_history`。
入口回放使用记录的代码及锁定依赖；候选必须自己完成当前原子的全部必要准备，不能只复用最后一次 repair 的中间状态。
旧经验缺少 trace 时仍用历史 prefix 规则；回放失败会区分 prefix、target、suffix，无法重建的旧经验不会污染正式回归库。
新 case 先进入 `pending_cases`，候选必须通过该 case 和所有**已独立回放通过**的正式案例，才会同事务发布版本并激活 case。失败的新 case 保留为待验证，不成为后续无关更新的必过门槛。
已发布技能在新任务中成功复用后也须独立回放，再激活新 case；成功复用不重复生成相同代码版本。
局部发布仅证明该原子的已验证效果，不保证整个任务。整任务成功的续接经验另存 `exploration/continuations/`；显式传给 `maintain --experience` 后才建立包含 suffix 和原生成功的更强回归义务。
旧库首次由新版打开时迁移到 schema 2：已发布 fresh replay 覆盖的 case 保留正式状态，其他 case 转待验证，已发布代码保留；旧冻结 worker 会拒绝新版库，需以新源码创建新 run。
这是一套经验回归机制：只覆盖已经登记的案例，不自动证明所有历史任务或未知场景都成功。

## 空槽位、代码和接口边界

`catalog.py` 定义 P01–P40 的名字、类别与建议参数；`library/stubs/Pxx.py` 起初全部抛出 `NotImplementedError`。
`init` 后 `active` 是空字典。目录里存在 P40 等 40 个定义，不代表已经实现 40 种物理能力。
P32–P39 中的柔性物折叠、撕离等行为不能仅凭当前 RoboCasa 厨房刚体环境获得覆盖，需要相应仿真任务和评估器。

```python
def skill(api, params):
    # 以下仅演示函数约定，不是操作技能或成功策略。
    observation = api.observe()
    api.wait(params["duration"])
    return {"observed_step": observation["step"]}
```

候选必须读取 `params`，返回诊断字典；不允许以返回 `success=True` 取代 host 判定。
目标、相机像素、位姿、方向和时长等场景绑定由计划中的 `params` 传入，不能把 seed 专用轨迹伪装成泛化代码。
规划应尽量直接给出可执行的相机/像素等绑定；运行中的 Python 没有 VLM，不能把“看图找到 Start”当作已实现的接口。
repair 模型看到新图后，可在候选的可选 `params` 字段返回**完整替换参数**；省略时沿用当前参数，不能改 atom、goal 或 postcondition。
代码继续从 `params` 读取新像素；参数变更记录来源。看不清时可先执行有界观察姿态调整，由下一次模型请求读取新图。
distill 接收原子入口观测。host 提供待验证 case 时，可额外返回完整 `params`，重新绑定该新 case 的入口像素/参数；生成新 case ID 并重新验证，不改动历史正式案例、goal 或 postcondition。
执行环境提供 `np`、`math` 及有限内置函数；代码不能导入模块、读写文件、调用网络、访问私有属性或修改模拟器。
AST 检查和时限是执行约束，不是操作系统安全沙箱；此框架面向受控研究 worker。

| 基础 API | 含义与单位 |
| --- | --- |
| `observe()` | instruction、RGB 路径、深度 `.npy`、相机标定、机器人测量和剩余 horizon 信息 |
| `step(action, repeat=1)` | `arm` 为 7 维绝对弧度；`torso` 为米；`gripper` 为 -1/0/+1；`base` 为机体坐标系 m/s、m/s、rad/s |
| `move_joints(target, duration=2, part="arm", ...)` | 普通关节插值运动，`part` 可为 `arm` 或 `torso` |
| `move_tcp(position, rotation=None, ...)` | 当前机器人基座坐标系中的米和 3×3 旋转矩阵；通用 IK，无物体语义控制器 |
| `move_base(velocity, duration=.5)` | 有时限的机体坐标系底盘速度，结束后停止 |
| `gripper(direction, duration=.5)` | -1 张开，+1 闭合；结束后保持原生夹爪预载 |
| `wait(duration=.5)` | 保持控制参考，推进物理仿真 |
| `deproject(camera, u, v, radius=1)` | RGB-D 像素反投影为当前机器人基座坐标中的三维点 |
| `call("Pxx", params)` | 仅调用已声明、已发布并锁定版本的依赖 |

完整签名以 `evo/runtime_api.py` 的 `API_DESCRIPTION` 为准。
RGB 图像会真正作为模型图片输入；原始深度由 `deproject` 使用，不把 `.npy` 路径假装成模型看过的深度图。
`semantic_catalog` 只给物体/设备标识、类别、别名及关节名，不给位姿或设备状态。
底层借用现有 RoboEnv 的 RoboCasa backend 和原生 controller，不携带其 70 个高级 `rc` 操作技能；没有预置 pick/place/press/turn 策略。

## 后验条件与观测隔离

每步格式为 `{id, atom, params, goal, postcondition, depends_on}`，依赖必须在前面或属于已完成前缀。
`postcondition` 为 `{"predicate":"名称","args":{...}}`，只允许 host 已声明的谓词。

| 谓词 | 检查内容 |
| --- | --- |
| `native_success` | 原生任务完成，只能用于最后一步 |
| `joint_near` / `tcp_near` | 机器人关节 / TCP 到达指定范围 |
| `gripper_aperture` | 夹爪开度，不能据此声称已经抓牢物体 |
| `object_grasped` | 原生双侧夹指接触；不等于持续稳定抓握 |
| `object_in_receptacle` / `object_in_fixture` | 原生容器接触/包容或设备包容关系 |
| `fixture_state` / `fixture_joint` | 已有设备布尔状态或归一化关节区间 |

语义 oracle 只存在于 host 评估阶段。返回的 verdict/reason 可用于下一轮模型失败反馈；隐藏的物体位姿、设备几何与真实状态不进入 `api.observe()`，生成函数也没有 `evaluate()` 接口。
未知设备字段或不支持的条件产生 `known=false`，不能记为成功。
后验条件必须真的对应原子的目标；“手到达附近”不等于“抓取成功”，模型计划仍需要研究者审查这一语义关系。

## 安装与测试

Python ≥3.10。核心依赖 NumPy；真实仿真还需要 RoboCasa、robosuite、MuJoCo、RoboEnv 及其 `roboenv_runtime`，以及资产、SciPy、Pillow 和可用渲染环境。这些仿真运行时和第三方资产不随本仓库发布。安装 Python 包并不等于完成仿真环境安装。

```bash
git clone https://github.com/ymzzzz-damn/evo-harness.git
cd evo-harness
python3 -m pip install -e '.[test,control]'
python3 -m pytest tests -q
python3 -m evo.cli init --library ./library
python3 -m evo.cli catalog
```

测试中的简单 toy plant 和诊断函数是测试夹具，不是已沉淀的机器人技能，也不代表 RoboCasa 成绩。

## 运行任务

```bash
cp configs/robocasa.example.json configs/robocasa.local.json
export OPENAI_API_KEY="YOUR_API_KEY"
# 编辑 local 配置中的 agent.base_url、model 和任务列表。
# 使用已安装 RoboCasa/RoboEnv 的 Python 环境执行：
python3 -m evo.cli run --config configs/robocasa.local.json --output runs/task-001
```

示例的 localhost 地址和 `YOUR_VISION_MODEL` 是占位配置，需要替换为实际支持图像输入的模型服务。密钥从 `api_key_env` 指定的环境变量读取，不写入配置或提交到仓库。运行输出目录必须不存在；技能库目录可以跨任务复用。

- `maintenance_trigger: "atomic"`：每个原子步骤成功立即维护。设为 `"episode"` 则每轮结束维护，包括失败轮。
- `maintain_after_task: false`：关闭自动维护；也可用 `explore` 子命令仅收集经验。
- `episode_retry.max_episodes: null`：同一 task 失败后持续 reset 重试；正整数限制 episode 数。每轮仍受原生 horizon、尝试次数和代码超时约束。
- `tasks`：任务名称、seed、相机尺寸等适配器配置。修改示例即可运行多个任务。
- `task_status` / `native_successes`：任务执行结果；`maintenance_status` / `promoted_versions` / `accepted_case_count`：技能发布与复用验收结果。

独立维护、检查和回滚：

```bash
python3 -m evo.cli maintain --config configs/robocasa.local.json --output runs/maintain-001 --experience /path/to/pending/01_P10.json
python3 -m evo.cli inspect --library ./library
python3 -m evo.cli rollback --library ./library P10 1
```

`evaluate` 子命令接收包含 `library` 和 `evaluations` 的配置，每项 evaluation 含 `task` 和已绑定参数的 `plan`。它冻结库版本，不修复代码、不写回技能库；需要先自行生成并验证技能，本仓库不附带成功策略。

## 源码布局

| 路径 | 用途 |
| --- | --- |
| `evo/catalog.py` | P01–P40 定义与空函数槽位 |
| `evo/agent.py`、`evo/prompts.py` | 规划、修复、蒸馏与模型接口 |
| `evo/loops.py` | 任务探索与独立回放维护 |
| `evo/library.py` | 候选、正式案例、版本和依赖锁定 |
| `evo/robocasa_adapter.py`、`evo/runtime_api.py` | 原生仿真适配与允许的控制接口 |
| `evo/codecheck.py`、`evo/execution.py` | 代码契约检查和有时限的执行 |
| `evo/cli.py` | 任务重试、即时维护和冻结评估入口 |
| `scripts/` | 部署、并行运行、资源测量、任务目录生成、恢复与统计工具 |
| `tests/` | 控制流、库事务、适配器及部署工具测试 |
| `configs/robocasa.example.json` | 不含凭据或已学技能的配置模板 |

`scripts/build_robocasa365_catalog.py` 可从实际安装的 RoboCasa 源码及其 manifest 构建任务目录；生成目录与 source manifest 不属于本次源码发布。

## P6000 部署脚本

`scripts/launch_p6000.py`、`launch_parallel_p6000.py` 和 `launch_gpu_parallel_p6000.py` 保留当前部署工具源码。它们依赖已有集群环境、外部 `runtime/deployment/job_template.json`、RoboEnv 运行时、仿真镜像和资产；脚本中的默认项目路径、运行时路径、kubeconfig、namespace 及 Python 路径需要按自己的环境配置。仓库不包含这些机器配置、凭据、镜像或第三方运行时。

launcher 为每次 run 冻结源码并记录哈希。GPU 并行入口先测量设备容量并保留利用率记录；单元测试不能替代真实环境下的资源与渲染验证。采集和统计脚本用于读取自行生成的运行目录，不依赖本仓库附带历史实验。

## 发布边界

`.gitignore` 排除技能库、生成的 `code_tools`、候选与经验记录、run 输出、observations/ 下的图像、深度/视频数据、日志、验证报告、仿真源码快照、缓存及本地凭据。整个实验专用 `scripts/kinex_eval50/` 不在本次发布范围内。开始新实验时，`init` 会生成空库；需要通过自己的探索和独立验证逐步积累能力。

局部能力发布不代表整个任务成功，单个 task/seed 回放成功也不证明跨任务泛化。已成功任务的 continuation 可另行送入维护，建立更强的完整流程回归约束。
