# DeepCAVE-X 作业任务备忘录

> 记录时间：2026-09-05
> 用户目标：基于 DeepCAVE 开发可视化扩展，并完成四页论文。

## 作业原文（浓缩）

AutoML 降低了机器学习应用门槛，但超参数优化过程本身是“黑箱”：

- 为什么选择这组参数？
- 搜索过程是否陷入局部最优？
- 哪些超参数最敏感？

DeepCAVE-X 不是一个新优化算法，而是一个可视化分析工具包：

1. 解析主流 AutoML 框架（SMAC、Optuna）产生的优化日志；
2. 将高维超参数搜索轨迹降维并交互式展示。

## 任务聚焦三点

1. **轨迹对比视图**：并排比较不同优化器（如贝叶斯优化 vs 随机搜索）的行为差异。
2. **超参数重要性分析**：基于随机森林或 Sobol 指数，量化各参数对最终性能的贡献。
3. **优化瓶颈诊断**：自动标记搜索过程中的性能停滞期，并建议调整策略。

## 入门路径

1. 通读 DeepCAVE 原论文及开源代码，理解架构。
2. 选取 2-3 个公开数据集（如 OpenML 分类任务），运行 SMAC 和 Optuna 生成优化轨迹。
3. 基于 DeepCAVE 的 API 开发上述扩展模块。
4. 编写四页论文：设计选择、使用案例、用户反馈（邀请 5-10 位同学试用并收集定性评价）。

## 声称的创新点

- 首次将多优化器轨迹对比功能集成到开源可视化工具；
- 提出动态瓶颈检测算法，实时标注优化过程中的低效探索区间；
- 开发轻量级 Web 交互界面，无需编程即可完成复杂诊断。

## 重要技术备注（需要和老师/文档核对）

- 当前 DeepCAVE v1.4.1 已经有 Run 选择、Group 联合分析以及多 run 的
  Cost over Time 对比；“DeepCAVE 仅支持单次运行分析”的说法不准确。
  创新点建议表述为“跨优化器的显式轨迹对比与诊断工作流”，而不是“填补完全空白”。
- 多个 run 对比要求 objectives / configspace / budgets 可合并，SMAC 与 Optuna
  结果需要先统一为可比较的格式。

## 本机已完成的环境

- Miniconda：`/Users/crystal/miniconda3`
- conda 环境：`deepcave`（Python 3.9.25，DeepCAVE 1.4.1，swig 4.4.1）
- Jupyter kernel：`DeepCAVE (3.9)`（已注册）
- conda/pip 源：清华 TUNA 镜像
- redis-server：已存在（v8.10.1），交互模式可用
- API 模式测试脚本：
  `/Users/crystal/Desktop/研一上/ML/deepcave-api-test/deepcave_api_test.py`

## 已踩坑记录

- DeepCAVE 1.4.1 的 `Recorder` 写出的 history 是 9 字段，但加载器按旧格式处理，
  导致 `DeepCAVERun.from_path` 失败；已在 env 内 `run.py` 打补丁，
  备份在 `/tmp/deepcave_run.py.bak`。重装/升级 deepcave 后需重新打补丁。
- DeepCAVE 固定 Plotly 5.24.1，静态出图需 `kaleido==0.2.1`（新版 1.4.0 不兼容）。
- Apple Silicon + Mac 上部分插件可能有 numpy/链接兼容问题，报错再排查。

## 下一步候选

- 选定 2-3 个 OpenML 数据集。
- 确定 SMAC3 与 Optuna 的搜索空间、预算、seed 方案，保证可比性。
- 设计 run 存储目录规范，使 DeepCAVE 能同时加载两组轨迹。
- 设计轨迹对齐指标（incumbent 曲线 / 平均最佳值 / 多样化指标等）。
- 开发跨优化器对比插件原型，再接入重要性分析与瓶颈诊断。
