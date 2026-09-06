# DeepCAVE-X 完整任务计划

> 依据作业任务 + DeepCAVE v1.4.1 现状制定。所有步骤可在云端 Linux 容器中执行，
> 本地 Mac 环境作为开发/备份环境。

## 0. 目标与交付物

**目标**：在 DeepCAVE 之上实现三个可视化/诊断能力，并用 SMAC 与 Optuna 的
真实轨迹完成案例研究与四页论文。

**交付物**

1. 可复现的云端容器开发环境（Dockerfile / devcontainer.json）。
2. SMAC + Optuna 优化轨迹数据集（2-3 个数据集，统一格式，纳入版本管理或网盘备份）。
3. DeepCAVE-X Python 扩展包 + 轻量 Web 交互界面。
4. 5-10 位同学的用户测试记录与定性反馈。
5. 四页论文。

## 0.5 与 DeepCAVE 的关系（重要）

**“基于 DeepCAVE 的 API 开发扩展模块” = 外围扩展，不是重写，也不是必须大改源码。**

推荐分层：

1. `deepcave` 作为已安装依赖，负责：run 加载（`DeepCAVERun.from_path`）、
   数据记录（`Recorder`）、轨迹/目标数据读取、现有 evaluator（fANOVA/LPI）。
2. 本项目 `deepcave_x` 作为独立扩展包，调用上述 API，实现：
   轨迹对比、重要性汇总、停滞检测、Web 界面。
3. 可选“插件级集成”：注册成 DeepCAVE 原生 Plugin 进入侧边栏，
   属于加分项而非主线。

不要 fork 重写 DeepCAVE；对 1.4.1 源码只保留必要的运行补丁（history 加载 bug），
并作为环境搭建步骤固化。

## 1. 任务分阶段计划

### P0 需求与事实确认（半天）

- 向老师确认：截止日期、四页论文模板（ACM/IEEE/其他）、是否需要提交代码仓库、
  评分重点（实现深度 vs 论文 vs 演示）。
- 精读 DeepCAVE 论文（arXiv 2206.03493）与文档，输出一页架构笔记。
- **校准创新点表述**：DeepCAVE 已有 Group / 多 run Cost over Time，
  不要写“填补仅支持单次运行的空白”，改写为
  “面向跨优化器的显式轨迹对比与停滞诊断工作流”。
- 产出：`deepcave-x-project.md` 更新为最终口径 + 任务检查清单。

### P1 云端容器环境（0.5-1 天）

- Ubuntu 22.04 + Miniconda，Python 3.9 环境 `deepcave`；
- 安装：`swig`、`redis-server`、`deepcave[optuna]`、SMAC3、ConfigSpace、
  OpenML、scikit-learn、ipykernel；
- DeepCAVE 1.4.1 已知问题处理：
  - 对 `run.py` 打 history 兼容补丁；
  - 固定 `kaleido==0.2.1`；
  - 不依赖 DeepCAVE 的 SMAC 转换器读新版 SMAC 日志（避免兼容问题）。
- 在容器内跑通 `deepcave-api-test/deepcave_api_test.py`。
- 产出：`devcontainer.json`、`Dockerfile`、一键环境脚本 `setup.sh`。

### P2 数据集与统一对比协议（1-2 天）

**选 2-3 个数据集**（小而稳定、可快速重复跑）：

| 优先级 | 数据集 | 规模特点 | 用途 |
|---|---|---|---|
| 1 | credit-g | ~1000 行，二分类 | 主案例 |
| 2 | vehicle | ~850 行，多分类 | 验证多分类 |
| 3 | segment / mfeat-factors | ~2300 行，多分类 | 增加难度梯度 |

- 若 OpenML 网络不稳定，fallback：sklearn 内置 `breast_cancer` / `digits` / `wine`。
- 数据缓存到容器持久卷，保证重复实验不重复下载。

**统一协议（可比性核心）**

- 同一分类器（默认 RandomForest；可选 MLP 作 case study）。
- 同一个 ConfigSpace（一份 YAML 定义，SMAC 与 Optuna 共用）。
- 同一目标：固定 stratified split 或 5-fold CV 的 `1 - balanced_accuracy`（最小化）。
- 每个 (dataset, optimizer, seed) 一个 DeepCAVE run：
  - 配置数量：开发期 30，正式实验 50-100；
  - seeds：至少 3-5 个；
  - 记录方式：在 optimizer 的 target function 中调用 DeepCAVE `Recorder`，
    统一写成原生 run，避免日志格式不兼容。
- 实验配置集中写在 `experiments/config.yaml`。

目录约定：

```text
experiments/
  datasets/
    credit-g/
      smac_seed0/run_1
      optuna-tpe_seed0/run_1
      optuna-random_seed0/run_1
      ...
```

### P3 核心模块开发（主体 5-7 天）

建议按“先 Python API，后 UI”的顺序开发，每个模块都先用小规模合成实验跑通。

#### 3.1 多优化器轨迹对比

- 工具函数：加载多 run、抽取每个 run 的 incumbent trajectory
  （按 trial 序号 / wall-clock 对齐）。
- 对比指标：
  - N 次试验后的 best-found；
  - regret 曲线 / 曲线下面积；
  - 达到某阈值的首次 trial；
  - 提升次数、无效试验占比。
- 输出：并排/叠加轨迹图、指标对比表。

#### 3.2 超参数重要性分析

- 复用 DeepCAVE 的 Importances / fANOVA 能力作为“RF 重要性”通道；
- 额外提供 Sobol 指数（stretch goal）：对连续型超参数空间做代理模型 + SALib
  或 Sobol 序列采样；categorical 处理方案写清楚；
- 输出：每个 run / optimizer 的重要性排序和对比条形图。

#### 3.3 优化瓶颈诊断

- 输入：单条或多条 incumbent trajectory；
- 停滞定义（可调参数）：
  - 连续 W 次试验无 improvement；
  - 或 improvement 小于阈值 ε；
  - 带 warm-up，避免误报初始探索期。
- 输出：
  - 停滞区间（起止 trial、持续时间、区间内试验数）；
  - 图上阴影标注；
  - 建议文本：结合重要性/预算/优化器类型给出
    （“该区间全为低预算评估”“高重要性参数仍在大范围采样”等）。

### P4 轻量 Web 界面（3-4 天）

根据 DeepCAVE 1.4.1 插件开发复杂度，优先实现**独立轻量 Dash 应用**：

- 读取 `experiments/` 下的 DeepCAVE runs（用 Python API，不侵入 DeepCAVE 内核）；
- 三个页面/面板：轨迹对比、重要性、瓶颈诊断；
- 参数下拉选择 dataset / optimizer / seed，全部免代码操作；
- 运行在云容器（默认 8050/8051），供同学访问。

如果时间充足，再尝试注册成 DeepCAVE 原生侧边栏插件（stretch goal）。

### P5 用户测试（2-3 天，可并行于论文写作）

- 招募 5-10 位同学；
- 给统一任务剧本（15-20 分钟）：
  1. 判断 SMAC 与随机搜索谁收敛更快、依据是什么；
  2. 找出最重要的 2 个超参数；
  3. 找出一次明显停滞并说出建议。
- 问卷：有用性/易理解性/愿意使用 Likert 题 + 开放式问题；
- 记录口头意见，至少迭代一轮 UI。

### P6 四页论文（4-5 天）

建议结构：

1. Introduction & Motivation（0.5-0.75 页）
2. Background & Design（1-1.25 页）：DeepCAVE 架构、三个模块设计选择
3. Case Study / Experiments（1-1.25 页）：数据集、对比协议、关键图
4. User Feedback & Conclusion（0.5 页）

必备图表：

- 多优化器 incumbent 轨迹对比图；
- 停滞区间标注示例；
- 重要性排序图；
- Web 界面截图。

引用：DeepCAVE 论文、fANOVA/LPI、Sobol、SMAC3、Optuna。

### P7 收尾

- README：环境启动、实验复现、Web 启动；
- 演示录屏或在线链接；
- 全量数据备份（云盘/网盘）；
- 按测试反馈做最终小迭代。

## 2. 时间线建议（无截止日期时的保守排期）

| 周 | 内容 | 里程碑 |
|---|---|---|
| 第 1 周 | P0-P2 | 容器就绪；数据集 + SMAC/Optuna 初版轨迹 |
| 第 2 周 | P3 | 三个核心模块 API 可用，含单元验证 |
| 第 3 周 | P4-P5 | Web 界面可用，完成第一轮用户测试 |
| 第 4 周 | P6-P7 | 论文定稿、演示、备份 |

## 3. 主要风险与对策

| 风险 | 对策 |
|---|---|
| DeepCAVE 1.4.1 依赖老旧、有已知 bug | 锁定 Python 3.9 + 依赖版本；记录并复现补丁 |
| SMAC 新版本与 DeepCAVE 转换器不兼容 | 用 Recorder 统一记录，转换器仅作备份方案 |
| OpenML 无法访问 | 用 sklearn 内置数据集，论文标注为“OpenML 风格” |
| 优化器跑太久 | 先用小搜索空间/30 trials 全链路验证，再扩量 |
| 同学无法访问容器 Web | 云平台开放端口或使用隧道，提供录屏兜底 |
| Dash 依赖冲突 | Web 层保持独立进程和独立 requirements |

## 4. 开工顺序

第一步：确认 P0（截止日期、模板、评分）；第二步：在云端容器跑通
`deepcave_api_test.py`；第三步：给出 2-3 个数据集候选并冻结实验协议。
