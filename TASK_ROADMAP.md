# 任务路线图（分步实现版）

> 目标：把当前“采集 trace → 构建数据集 → 评估/对比”的工具链，拆成可逐步落地的开发任务，便于你按步骤写代码并在每一步拿到可验证产物。

## 0. 使用方式（建议）

- 按顺序完成下面每个阶段，不跳步。
- 每个阶段都包含：**要写什么**、**完成标准**、**最小验证命令**。
- 建议每完成 1~2 个阶段就提交一次小 commit，方便回滚和评审。

---

## 阶段 1：统一数据结构与 I/O 基座

### 你要写什么
1. 在 `core_types.py` 固化最小核心对象：
   - theorem/task 信息（如 repo、file、full_name）
   - transition 信息（state_before、action、state_after、reward/done）
2. 在 `trace_io.py` 固化 JSONL 读写工具：
   - append 写入
   - 迭代读取（流式）
   - 基本字段完整性检查

### 完成标准
- 任何脚本都能通过同一套函数读写 trace。
- trace 至少包含：`run_id`、`theorem`、`transitions`、`timestamp`（或等价字段）。

### 最小验证命令
- `python -c "from trace_io import iter_jsonl; print('ok')"`

---

## 阶段 2：实验目录规范（可复现最小闭环）

### 你要写什么
1. 在 `experiment_io.py` 统一 run 目录约定：
   - `runs/<run_id>/config.json`
   - `runs/<run_id>/traces.jsonl`
   - `runs/<run_id>/metrics.json`
2. 提供函数：
   - 创建 run 目录
   - 落盘配置
   - 落盘指标

### 完成标准
- 采集脚本和评估脚本都不再自行拼路径，统一调用 `experiment_io.py`。
- 同一 `run_id` 下能复现输入配置和输出指标。

### 最小验证命令
- `python -c "from experiment_io import make_run_dir; print('ok')"`

---

## 阶段 3：环境适配与容错（避免全流程中断）

### 你要写什么
1. 在 `env.py` 提供 LeanDojo 交互薄封装（如 `run_transition`）。
2. 明确“可跳过”与“必须失败”两类错误：
   - 缺少 artifact：默认 skip + warning
   - 指定 `--fail-on-skip`：立即失败
3. 为 skip 记录原因字段，便于统计。

### 完成标准
- 批量任务中单 theorem 失败不会直接拖垮全流程。
- 日志中可追踪每个 skip 的原因。

### 最小验证命令
- `python search_generate_traces.py --help`

---

## 阶段 4：任务注册与基准配置

### 你要写什么
1. 在 `tasks.py` 维护任务注册表（task id → theorem 列表/生成器）。
2. 在 `benchmark_specs.py` 维护可复现实验规格：
   - 任务集
   - 采样数量/步数上限
   - 随机种子

### 完成标准
- 新增 benchmark 只需改注册表和 spec，不需改主流程脚本。
- 通过 spec 可一键复跑同一实验。

### 最小验证命令
- `python -c "from benchmark_specs import BENCHMARK_SPECS; print(list(BENCHMARK_SPECS)[:3])"`

---

## 阶段 5：采集脚本标准化（单例 + 批量）

### 你要写什么
1. `collect_one_example.py`：单 theorem 采集，便于调试。
2. `collect_traces.py` / `search_generate_traces.py`：批量采集。
3. CLI 统一参数命名：输入任务、输出 run_id、max_steps、seed。

### 完成标准
- 单例模式可快速复现 bug。
- 批量模式可稳定输出 `traces.jsonl`。

### 最小验证命令
- `python collect_one_example.py --help`
- `python collect_traces.py --help`

---

## 阶段 6：数据集构建（SFT 可直接消费）

### 你要写什么
1. `build_sft_dataset.py` 以流式方式读取 trace（避免大文件爆内存）。
2. 统一 prompt 构造入口（如 `build_prompt`），避免多个脚本各写一套。
3. 可选写入 metadata（run_id、theorem id、step id）。

### 完成标准
- 输出 JSONL 可直接给训练脚本使用。
- 同一 trace 输入在固定参数下输出确定。

### 最小验证命令
- `python build_sft_dataset.py --help`

---

## 阶段 7：评估与对比（形成可汇报结果）

### 你要写什么
1. `evaluate_traces.py` 产出基础指标：
   - 样本数
   - 平均步数
   - 成功率（若定义）
   - skip 统计
2. `compare_runs.py` 对两个或多个 run 的 `metrics.json` 做并排比较。

### 完成标准
- 每次实验都可自动得到 `metrics.json`。
- 比较结果可直接用于 PR 描述或实验日志。

### 最小验证命令
- `python evaluate_traces.py --help`
- `python compare_runs.py --help`

---

## 阶段 8：总控 pipeline（把“手工步骤”变“一条命令”）

### 你要写什么
1. `run_pipeline.py` 编排全链路：
   - collect/search → build_sft → evaluate → compare（可选）
2. 加 `--dry-run`：只打印将执行的命令，不实际运行。
3. 加前置依赖检查与清晰报错（缺命令/缺模型/缺目录）。

### 完成标准
- 新同学按 README 一条命令能走完最小闭环。
- `--dry-run` 可作为 CI 中“编排正确性”检查。

### 最小验证命令
- `python run_pipeline.py --pipeline classifier --dry-run`

---

## 阶段 9：文档与验收清单（避免“能跑但不可维护”）

### 你要写什么
1. 文档保持三件套：
   - `BENCHMARK_WORKFLOW.md`（流程图 + 命令）
   - `TRACE_SCHEMA.md`（字段约定）
   - `REPRO.md`（复现约束、随机种子、环境）
2. 每个 CLI 文档至少给一个最小示例。

### 完成标准
- 文档命令可复制执行。
- 新增字段时，schema 文档同步更新。

### 最小验证命令
- `python run_pipeline.py --help`

---

## 建议的 2 周节奏（可选）

- **第 1 周**：阶段 1~5（把数据结构、I/O、采集链稳定下来）
- **第 2 周**：阶段 6~9（补齐训练输入、评估对比、总控和文档）

---

## 你可以直接照着做的“每次开发循环”

1. 先选一个阶段。
2. 只改该阶段相关文件。
3. 跑该阶段的最小验证命令。
4. 记录“输入参数 + 输出文件 + 指标”。
5. 提交小 commit。

如果你愿意，我下一步可以按这个路线图从**阶段 1**开始，直接给你“第一步具体要改哪几行”的最小补丁计划。
