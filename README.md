# DeepCAVE-X（deepcave-experiments）

**DeepCAVE-X** is a decision-centered visual analytics framework for hyperparameter optimization.
It treats an HPO run not merely as a collection of trials to be visualized, but as a **sequential decision-making process**.
DeepCAVE-X organizes diagnostic evidence into four layers: **performance sufficiency, parameter attribution, search behavior, and resource fidelity**.
By linking visualization evidence to concrete actions—stop, continue, expand, contract, reparameterize, or adjust fidelity—DeepCAVE-X closes the loop between visual analysis and HPO decision-making.

> **中文简介**：DeepCAVE-X 是一个以决策为中心的超参数优化（HPO）可视化分析框架。它把一次 HPO 运行看作一个序列决策过程，而不仅仅是待可视化的试验集合；将诊断证据组织为"性能充分性、参数归因、搜索行为、资源保真度"四个层面，并把可视化证据与具体行动（停止 / 继续 / 扩展 / 收缩 / 重参数化 / 调整保真度）关联起来，打通"可视化分析 → HPO 决策"的闭环。

本仓库基于 **DeepCAVE v1.4.1**（[automl/DeepCAVE](https://github.com/automl/DeepCAVE)），在**保留其全部原有功能**的基础上，新增了实验性诊断能力

## 1. 项目结构

```
deepcave-experiments/
├── deepcave/                     # 主包（DeepCAVE-X，v1.4.1）
│   ├── __init__.py               #   应用初始化、Web/API 双模式判定
│   ├── cli.py                    #   `deepcave` 命令行入口（absl flags）
│   ├── start.sh                  #   启动脚本：redis → worker → server
│   ├── server.py / worker.py / open.py
│   ├── config.py                 #   配置类：端口、目录、插件/转换器注册表
│   ├── constants.py / custom_queue.py
│   ├── plugins/                  # 【插件层】Dash 交互展示（仅 Web 模式生效）
│   │   ├── static.py             #   StaticPlugin 基类（重计算走 RQ 队列+缓存）
│   │   ├── dynamic.py            #   DynamicPlugin 基类（轻量计算，回调内直接执行）
│   │   ├── summary/              #   Overview / Configurations / Footprint / Footprint Replay★
│   │   ├── objective/            #   Cost over Time / Pareto Front
│   │   ├── budget/               #   Budget Correlation
│   │   └── hyperparameter/       #   Importances(含★Weighted Sobol) / Ablation Paths /
│   │   │                         #   Configuration Cube / Parallel Coordinates /
│   │   │                         #   Partial Dependencies / Symbolic Explanations /
│   │   │                         #   Sampling Density★
│   ├── evaluators/               # 【算法层】纯 Python 计算，Web 与 API 双入口通用
│   │   ├── fanova.py / lpi.py / ablation.py / footprint.py
│   │   ├── mo_fanova.py / mo_lpi.py / mo_ablation.py   # 多目标变体
│   │   ├── weighted_sobol.py★ / sampling_density.py★ / footprint_replay.py★
│   │   └── epm/                  #   代理模型（pyrfr 随机森林、pyPDP 兼容代理）
│   ├── runs/                     # 【数据层】Run/Group/Trial/Objective/Recorder
│   │   └── converters/           #   smac3v1 / smac3v2 / bohb / optuna / amltk /
│   │                             #   raytune / dataframe / deepcave（原生）
│   ├── layouts/                  #   页面布局（侧边栏、页头、主区域、通知）
│   └── utils/                    #   工具（布局、压缩、日志、样式化绘图等）
├── configs/                      # 配置模板：local.py（DEBUG）/ server.py（生产）
├── logs/                         # 预置示例运行（见 §8）
├── examples/                     # 示例：api/（API 用法）、record/（Recorder 记录）
├── tests/                        # 单元测试：test_evaluators / test_runs / test_utils
├── docs/                         # Sphinx 文档源（含各插件 rst 说明）
├── development-plan.md           # DeepCAVE-X 插件开发计划（中文，v1.1）
├── proposal.md                   # DeepCAVE-X 选题报告（中文，v1.1）
├── CHANGELOG.md / Makefile / setup.py / requirements.txt / pyproject.toml
└── README.md                     # 本文件（★ = 本仓库相对原版 DeepCAVE 新增）
```

## 2. 环境要求

| 依赖 | 要求 | 说明 |
|------|------|------|
| Python | **3.9 或 3.10**（`>=3.9, <3.11`） | 版本过高/过低均无法安装 |
| 操作系统 | Linux（官方仅正式支持） | macOS 上 fANOVA/`pyrfr` 存在已知兼容问题 |
| redis-server | 必需 | Web 模式的任务队列（默认端口 6379） |
| swig | 编译 `pyrfr` 所需 | `conda install -c anaconda swig` 或 `apt install swig` |
| Python 包 | 见 `requirements.txt` | ConfigSpace 1.2.0、dash 2.0.0、pyrfr、plotly、rq 等，安装时自动解决 |

## 3. 安装（从仓库路径 pip install，已验证）

```bash
# 1) 创建并激活 conda 环境
conda create -n DeepCAVE python=3.9 -y
conda activate DeepCAVE

# 2) 安装系统依赖（Ubuntu/Debian；已有可跳过）
apt-get install -y redis-server swig

# 3) 从本地仓库路径安装（editable 模式：修改源码即时生效）
pip install -e /root/deepcave-experiments
```

安装成功后，`deepcave` 命令会被注册到环境 `bin/` 下（`setup.py` 的 `console_scripts` 入口 → `deepcave.cli:main`）。

## 4. 启动与使用

### 4.1 Web 版（交互式仪表盘）

```bash
conda activate DeepCAVE
deepcave --open            # 启动并自动在浏览器打开（默认 1 个 worker）
deepcave --n_workers=2     # 指定后台 worker 数（StaticPlugin 重计算并行度）
deepcave --config server   # 使用 configs/server.py 配置（生产、非 DEBUG）
```

`deepcave` 命令经由 `deepcave/start.sh` 依次完成：

1. 检查/启动 **redis-server**（默认 6379 端口，任务队列）；
2. 启动 N 个 **RQ worker**（`worker.py`，异步执行 StaticPlugin 的重计算并缓存结果）；
3. 启动 **Dash Web 服务器**（`server.py`，默认 **http://127.0.0.1:8050/**）。

> **远程服务器访问**：DeepCAVE 默认监听 `127.0.0.1:8050`。在本地机器执行
> `ssh -L 8050:127.0.0.1:8050 user@server` 建立端口转发后，即可在本机浏览器打开
> http://127.0.0.1:8050 。也可以自定义 `configs/*.py` 中的 `DASH_ADDRESS`/`DASH_PORT`，
> 再用 `--config` 传入。

### 4.2 API 版（脚本化分析，无需启动 GUI）

DeepCAVE 检测到非 `server.py/worker.py` 执行时自动进入 API 模式，evaluator 算法层可直接调用：

```python
from deepcave.runs.converters.deepcave import Run  # 或 smac3v2/optuna 等转换器
from deepcave.evaluators.fanova import fANOVA
from deepcave.evaluators.weighted_sobol import calculate as weighted_sobol
from deepcave.evaluators.footprint_replay import build_cloud, subset

run = Run.from_path("logs/DeepCAVE/minimal/run_1")

# fANOVA 全局重要性
ev = fANOVA(run)
ev.calculate(run.get_objective(0), budget=run.get_highest_budget())
print(ev.get_importances())
```


## 5. 致谢与引用

本仓库基于 [automl/DeepCAVE](https://github.com/automl/deepcave) v1.4.1 二次开发，新增功能的设计动机详见仓库根目录的 `development-plan.md` 与 `proposal.md`。

若使用 DeepCAVE，请引用其 [ReALML@ICML'22 workshop 论文](https://arxiv.org/abs/2206.03493)：

```bibtex
@misc{sass-realml2022,
    title = {DeepCAVE: An Interactive Analysis Tool for Automated Machine Learning},
    author = {Sass, René and Bergman, Eddie and Biedenkapp, André and Hutter, Frank and Lindauer, Marius},
    doi = {10.48550/ARXIV.2206.03493},
    url = {https://arxiv.org/abs/2206.03493},
    publisher = {arXiv},
    year = {2022},
    copyright = {arXiv perpetual, non-exclusive license}
}
```

Copyright (C) 2021-2024 The DeepCAVE Authors · Apache License 2.0




