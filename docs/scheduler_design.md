# 调度系统设计方案（建议）

本文以现有实现为基础，给出一个“可落地、可演进”的调度系统设计建议，重点解决：**如何获得可靠的资源监控数据**、**如何把监控数据反馈到调度策略**、以及**如何用事件总线/存储形成闭环**。

## 0. 现状（代码资产）

- **系统/Agent 资源采样**：`backend/monitoring/system_metrics.py::SystemMetricsCollector` 周期性发布 `agent.resource` 事件。
- **全局自适应调度**：`backend/monitoring/resource_scheduler.py::ResourceScheduler` 订阅 `agent.resource` / `environment.alert`，动态调整 `global_workspace` 注意力阈值、模块轮询周期，并可调节 `backend/execution/scheduler.py::Scheduler` 的权重。
- **任务执行与队列**：`backend/execution/task_manager.py::TaskManager` 提供优先级/截止期/设备的队列与执行，发布 `task_manager.*` 事件。
- **环境侧信号**：`modules/environment/environment_adapter.py::EnvironmentAdapter` 可采集 CPU/内存负载并通过 registry/event bus 上报资源信号（更偏“环境自适应”）。
- **边缘设备探测**：`backend/edge/resource_manager.py::EdgeResourceManager` 提供 CPU/内存/GPU 可用性探测（适合 edge 场景）。

## 1. 资源使用情况的收集与监控

要实现“调度”，必须先有 **统一、实时、结构化** 的资源数据。建议把资源监控拆成两层：**系统层监控** 与 **任务层监控**，并通过事件总线汇聚。

### 1.1 系统层监控（System-level）

目标：持续、低开销地给出系统/进程层面的 CPU/内存（必要时扩展 GPU）负载，用于调度的“当前态”判断。

建议：

- 继续以 `SystemMetricsCollector` 为主：对各 Agent/进程周期采样，发布 `agent.resource` 事件。
- 采样频率可配置（例如 5 秒一次）：在保证调度反馈速度的同时控制开销，避免高频采样引起抖动。
- 指标建议最小集合：`cpu_percent`、`memory_percent`；如有条件再加 `gpu_utilization`、`gpu_mem_used`、`temperature`（可选依赖）。

事件建议（示例）：

```json
{
  "agent": "alpha",
  "cpu": 73.2,
  "memory": 55.1
}
```

### 1.2 任务层监控（Task-level）

目标：让调度系统能“感知单个任务”的资源开销，支持：

- 任务类型/类别的成本画像（CPU 密集、内存波动大、耗时长等）
- 后续做**资源预测**、**并发控制**、**设备选择（cpu/gpu）**、**限流/降级**等策略

建议做法（渐进式）：

1) **轻量估算（优先落地）**：在 `TaskManager` 任务开始/结束时记录运行时信息（耗时、CPU time、RSS 变化），随 `task_manager.task_completed` 事件上报。

现已在 `backend/execution/task_manager.py` 中给 `task_manager.task_completed` 增加了 `runtime` 字段（CPU time/CPU%/RSS delta 等），用于后续分析与调度决策。

2) **更精确计量（可选演进）**：

- 线程/进程 CPU 时间：线程池场景可用 `thread_time`；多进程/分布式可用进程级 CPU 时间。
- GPU 显存/利用率：可引入 NVML（如 `pynvml`）或框架 API（如 `torch.cuda`）做 best-effort 采样。

事件建议（示例）：

```json
{
  "task_id": "…",
  "name": "vectorize",
  "category": "embedding",
  "status": "completed",
  "device": "cpu",
  "duration_s": 0.83,
  "runtime": {
    "cpu_time_s": 0.62,
    "cpu_percent": 74.6,
    "rss_start_bytes": 123456789,
    "rss_end_bytes": 130000000,
    "rss_delta_bytes": 6543211
  }
}
```

## 2. 统一资源监控服务（Resource Telemetry Service）

为避免“各处采集、各处上报、口径不一”，建议在架构上明确一个统一入口：

- 统一采集：系统层（周期采样） + 任务层（任务埋点）
- 统一事件：通过 EventBus 发布（便于解耦、便于横向扩展）
- 统一落库：复用 `backend/monitoring/metrics_collector.py::MetricsCollector` / `TimeSeriesStorage`，形成可查询的时序数据

最终你会得到：**实时事件流（调度用）** + **历史时序库（预测/分析用）**。

### 2.1 调度输入快照（总结）

通过上述监控手段，调度系统将掌握全面的资源使用快照：包括整体 CPU/内存/GPU 状况、各 Agent/任务的资源占用，以及（在分布式情况下）各节点的负载情况。这些数据会作为后续调度决策的基础输入（如准入控制、权重自适应、并发/降级策略、设备选择等）。

## 3. 调度目标与约束（明确“优化什么”）

建议先把目标写清楚，避免后续策略互相打架：

- **稳定性优先**：避免资源打满导致整体退化（超时、OOM、响应变慢）。
- **吞吐与时延平衡**：在负载高时自动降并发/降级；负载低时提高并发。
- **公平性**：避免某一类任务/某个 Agent 长期饥饿。
- **可解释/可回滚**：调度决策要能被观测、可配置、可禁用。

## 4. 调度策略（建议从“启发式+反馈”开始）

结合现有实现，建议以“反馈控制”形式演进：

- **Admission Control（准入控制）**：当 `agent.resource` 持续高于阈值时，对新任务做排队/延迟/降级。
- **Queue + Priority**：继续使用 `TaskManager` 的优先级/截止期队列；对关键链路（CRITICAL/HIGH）保证及时性。
- **权重自适应**：复用 `ResourceScheduler` 调整 `Scheduler.set_weights(cpu=…, memory=…, tasks=…)`，把“系统负载”和“队列压力”映射成调度偏好。
- **模块轮询自适应**：继续使用 `ResourceScheduler.register_module()` 对传感/监听类模块做减速/加速，避免高负载时“监控本身成为负担”。

## 5. 预测与自适应（下一阶段）

在有了任务层 `runtime` 数据后，可以做轻量预测：

- 按 `category`/`name` 聚合 `duration_s` / `cpu_time_s` / `rss_delta_bytes`
- 使用 EMA（指数滑动平均）更新成本画像
- 调度时用画像做：
  - 设备选择（cpu/gpu）
  - 并发配额（同类任务限流）
  - 截止期/优先级动态提升（避免超时）

## 6. 运维与可观测性

建议至少保证：

- 所有调度关键事件都可追踪：`agent.resource`、`task_manager.task_*`、`monitoring.scheduler`
- 关键阈值可配置（env/config），并提供“一键关闭自适应”的开关
- 高负载时的降级策略可验证（例如：降低并发、降低轮询频率、切换轻量模型）

## 7. 多节点监控（分布式部署）

当系统未来以多节点/多 worker 方式部署时，建议把“监控采集”与“全局视图/调度”拆开：

- **节点侧（Agent/Worker）**：每个节点运行一个轻量监控代理（可复用 `EnvironmentAdapter` 或独立采集器），以 `worker_id` 标识节点/进程，并定期上报资源信号。
- **中心侧（Coordinator/Scheduler）**：订阅集群事件流（或从中心库拉取），把各节点上报的指标汇聚成全局视图，再把结果反馈给调度策略。

### 7.1 充分利用 `HardwareEnvironmentRegistry`

`modules/environment/registry.py::HardwareEnvironmentRegistry` 已具备多 worker 的能力模型与“最新指标”存储结构，建议作为中心侧的“当前态”事实来源：

- 节点启动：`get_hardware_registry().register(worker_id, capabilities, metadata=...)`
- 节点运行：周期调用 `report_resource_signal(worker_id, resource_signal, event_bus=cluster_bus)` 发布 `resource.signal`
- 中心侧：订阅 `resource.signal` 并将其写入中心侧 registry（形成全局视图，支持 `snapshot()`）

建议上报字段遵循统一口径（示例）：

- `cpu_percent` / `memory_percent`：百分比口径，调度更直观
- `cpu_utilization` / `memory_utilization`：0~1 归一化口径，便于算法处理
- `queue_depth`：队列深度（已在 `TaskManager` 的资源信号中体现）

### 7.2 集群事件总线/中心库

实现“多节点上报 -> 中心汇聚”有两条路：

1) **集群版事件总线**：例如 Redis Pub/Sub（`modules/events/redis_bus.py::RedisEventBus`），节点 publish，中心 subscribe。
2) **中心数据库汇总**：节点将指标写入中心 TSDB/SQL，中心按时间窗口查询汇总。

调度闭环的关键是：中心侧能以低延迟获得“最新态”，同时能查询短期历史用于趋势判断。

## 8. 历史记录与趋势（滑动窗口）

单点瞬时采样不够支撑鲁棒调度：需要能区分“瞬时峰值”与“持续压力”。建议对资源数据建立短期历史窗口（例如 1 分钟 / 5 分钟），并基于窗口统计触发不同策略。

### 8.1 窗口指标建议

- **均值/EMA**：用于判断持续负载（抑制尖峰噪声）
- **峰值/p95**：用于识别突发尖峰（但不一定触发强制限流）
- **持续时间**：例如“连续 30 秒均值 > 阈值”才触发干预

`EnvironmentAdapter` 已通过 `_samples` 队列实现“持续采样 + 平均负载”的思路；调度侧建议同样维护窗口（可按 worker/agent 维度分别维护）。

### 8.2 策略分流（瞬时 vs 持续）

典型建议：

- **瞬时峰值**（短窗口峰值高、均值不高）：倾向“不强制限流”，而是降低非关键模块轮询、延迟低优先级任务。
- **持续高负载**（窗口均值持续高）：触发“主动干预”，例如降并发、暂停非关键任务、切换轻量模型/执行模式，或发布 `environment.alert` 让 `ResourceScheduler` 进入保守模式。

### 8.3 实现方式（两种都可）

- **内存滑动窗口**：中心侧维护每个 worker 的 ring-buffer/deque（低延迟，适合实时调度）。
- **时序落库查询**：用 `TimeSeriesStorage` / TSDB 按 `start_ts` 查询最近 N 秒事件（实现简单，适合离线分析与报表；实时性取决于写入/查询开销）。

## 9. 任务/模块的调度策略与优先级设计

有了资源信息后，下一步是定义调度策略：如何根据资源状况与任务特性决定**执行顺序**、**执行地点（节点/Agent/设备）**与**执行方式（并发/降级/延迟）**。建议优先采用“可解释、可配置”的启发式策略，并逐步演进为数据驱动策略。

### 9.1 任务优先级与截止期（TaskPriority + Deadline）

项目已有 `backend/execution/task_manager.py::TaskPriority` 与 `TaskManager.submit(priority=…, deadline=…)`：

- **优先级（Priority）**：用于跨类别任务的粗粒度排序（CRITICAL > HIGH > NORMAL > LOW）。
- **截止期（Deadline）**：用于同优先级内的细粒度排序（更早的 deadline 先执行）；也可表达 SLA（如“必须 2s 内完成”）。

建议原则：

- Priority 决定“在资源紧张时谁先活下来”；Deadline 决定“同一生存层级里谁更急”。
- Deadline 不要滥用成“强制超时”，它更适合作为排序信号；真正的超时应由调用方/任务本身控制。

### 9.2 任务优先级赋值规则（提交时确定）

关键在于：任务在**生成/提交**时确定优先级（并在必要时设置 deadline / device / metadata）。建议建立统一规则（按来源、类别、时效）：

- **交互/关键链路（HIGH/CRITICAL）**
  - 用户交互请求、关键决策/执行链路、故障恢复/自修复、健康检查的关键动作：`HIGH` 或 `CRITICAL`
  - 建议同时设置较短 deadline（例如 now + 1~10s，视 SLA 而定）
- **常规前台（NORMAL）**
  - 与当前目标推进相关但非“立即响应”的任务：`NORMAL`
  - deadline 可不设置或设置为较宽窗口
- **后台维护（LOW）**
  - 日志归档、低频指标汇总、AutoML/学习等“可延迟”工作：`LOW`
  - 建议设置较宽 deadline（例如 now + 5~30min），或完全交由后台管理器在合适时机提交

落地建议：

- 统一使用 `category`/`metadata["source"]` 作为规则入口（便于后续统计与回放）。
- 针对关键类别明确默认 SLA：例如 `planning/decision` 默认 HIGH + 10s；`learning/automl` 默认 LOW + 300s。

### 9.3 动态调整：过载时的“升/降级”与“暂停”

当系统进入高负载/过载状态时，不建议在队列里“强行改优先级”（实现复杂且容易引入竞态）。更推荐两类策略：

1) **准入控制（Admission Control）**：在提交阶段直接延后/拒绝低价值任务
   - 例如持续高负载时：暂停提交 `LOW` 类后台任务（学习/AutoML/离线评估）
   - 资源恢复后再恢复提交（或按照 token/配额逐步放行）

2) **执行方式调整（Degrade/Throttle）**：保持关键链路吞吐
   - 降低后台模块轮询频率（复用 `ResourceScheduler.register_module()`）
   - 降低后台并发（例如学习/AutoML 令牌桶已在 `LearningManager` 中体现）
   - 选择更轻量的执行模式/模型（可由 `EnvironmentAdapter` 给出建议）

建议将“过载状态”判定做成**可配置阈值 + 短期窗口**（见第 8 节），避免瞬时峰值导致频繁抖动：

- 瞬时峰值：倾向只做轻量降噪（减慢非关键模块），不强行限流
- 持续压力：触发强干预（暂停 LOW 类、降低并发、切换轻量方案），并可发布 `environment.alert` 让 `ResourceScheduler` 进入保守策略

### 9.4 执行地点与设备选择（节点/Agent/CPU-GPU）

优先级决定“谁先跑”，还需要决定“在哪里跑”：

- **多 Agent（同节点）**：`backend/execution/scheduler.py::Scheduler` 会基于 `cpu/memory/tasks` 选择最空闲 Agent；`ResourceScheduler` 可动态调整权重以偏向某种资源。
- **多节点（分布式）**：建议以 `resource.signal` + `HardwareEnvironmentRegistry.snapshot()` 形成全局 worker 视图，再按“最小负载/最合适能力”挑选节点。
- **设备（cpu/gpu）**：对 GPU 任务显式传 `device="gpu"`；若不确定，可先在 metadata 标注“prefer_gpu/allow_cpu_fallback”，由上层策略决定。

### 9.5 模块调度：轮询优先级与响应性

“模块”通常表现为周期性轮询/监听/感知组件。建议沿用 `ResourceScheduler` 的做法，把模块分为三类并配置不同的 `slowdown_factor/boost_factor`：

- **关键传感/告警**：负载高时也要保证响应（boost_factor 较高）
- **常规感知/规划**：随负载自适应（中等系数）
- **后台维护/统计**：负载高时优先降速甚至暂停（slowdown_factor 较高）

这样在高负载时，系统能把资源集中到关键链路，同时避免“监控/轮询本身成为负担”。

### 9.6 动态并发控制（基于持续负载）

当平均 CPU 使用率在短期窗口内持续超过阈值（例如 >85%），可以降低同时运行的任务数量来缓解压力；负载恢复后再逐步恢复并发。建议优先复用两类现有能力：

- **持续负载判定**：沿用 `EnvironmentAdapter` 的采样窗口与 `desired_conc` 计算思路（避免被瞬时尖峰误触发）。
- **并发门控实现**：利用 `TaskManager` 每个 device 内部的并发 gate（信号量语义）做限流，而不必频繁重建线程池。

落地建议：

- 对 CPU 设备动态限流：`TaskManager.set_device_concurrency_limit("cpu", desired_conc)`（会自动 clamp 到 `[1, cpu_max_workers]`）。
- 过载时减半、恢复时回升：可以用 `desired_conc = round(base_max * 0.5)` 这类简单规则先落地，再用历史画像做更精细的类别限流。

工程实现备注：

- 本仓库已在 `TaskManager` 内实现“可调并发 limiter”，可在运行时调整 device 的并发上限（不重建 executor）。
- 若希望由 `EnvironmentAdapter` 自动驱动 CPU 并发调整，可在运行时启用 `TASK_MANAGER_DYNAMIC_CONCURRENCY=1` 并同时启用 `ENVIRONMENT_ADAPTER_ENABLED=1`（`EnvironmentAdapter` 将计算 `desired_conc` 并反馈给 `TaskManager`）。

### 9.7 跨设备调度（CPU vs GPU）

对于“既可 CPU 也可 GPU 执行”的任务（如推理/向量化/部分模型计算），建议在提交阶段做设备选择，以优先利用 GPU、同时避免 GPU 成为系统瓶颈：

- **GPU 空闲且未过载**：优先投递到 `device="gpu"`（加速完成、降低 CPU 压力）。
- **GPU 过载或排队严重**：若任务允许 CPU 回退，则改投递到 `device="cpu"`；若任务必须使用 GPU，则保持 `device="gpu"` 并排队等待。

落地方式建议以“任务标签/属性”驱动：

- `metadata["gpu_capable"]=True`：表示可在 GPU 执行
- `metadata["gpu_required"]=True`：表示必须 GPU（不应回退）
- `metadata["allow_cpu_fallback"]=True/False`：GPU 忙时是否允许回退到 CPU

工程实现建议：

- 提交时可使用 `device="auto"` 让 `TaskManager` 根据 GPU 可用性与负载阈值自动路由到 `cpu/gpu`。
- GPU 过载阈值可配置（例如 `TASK_MANAGER_GPU_OVERLOAD_THRESHOLD=0.9`）。实现上可优先采用 best-effort 指标（如显存压力 used/total）作为“过载”代理；后续有 NVML 等依赖时再升级为利用率+显存的组合判定。
- 为避免“GPU 队列阻塞导致 CPU 任务也无法调度”的头部阻塞问题，可在 dispatcher 中引入扫描/跳过机制：当某设备无可用并发槽位时暂存该任务、继续寻找可调度任务（扫描深度可配置，如 `TASK_MANAGER_DISPATCH_SCAN_LIMIT`）。

### 9.8 模块加载策略（按需加载 + 惰性卸载）

对于插件/工具类 capability 模块，建议把“加载/卸载”也纳入调度闭环：**新计划到来时按需预热**，**空闲时惰性回收**，并通过事件流形成可观测闭环。

落地建议：

- **按需预热（Plan/Directive）**：在 `planner.plan_ready` 事件中通过 `goal`/`metadata` 的 `[capabilities:...]`（或显式 `required_modules` 字段）标注所需模块；`AgentLifecycleManager` 收到后调用 `RuntimeModuleManager.ensure()/update()` 确保模块加载就绪。
  - 若希望“立即卸载无关模块”，可启用 `MODULE_MANAGER_PRUNE_ON_PLAN_READY=1`；否则默认只加载缺失模块，卸载交给生命周期组件处理。
- **使用记录（last used）**：`RuntimeModuleManager` 在模块首次加载时发布 `module.loaded`，并在每次命中/使用时发布 `module.used`；生命周期组件据此维护 `last_used_ts`，区分“刚加载但没再用”与“持续活跃”。
- **惰性卸载（Idle reclaim）**：`ModuleLifecycleManager.evaluate()` 周期运行（可由健康监控 tick 驱动），当模块闲置超过阈值时发布 `module.lifecycle.suggest_unload`，并在 `MODULE_LIFECYCLE_AUTO_UNLOAD=1` 时自动卸载回收内存。
  - 典型配置：`MODULE_LIFECYCLE_ENABLED=1` + `MODULE_LIFECYCLE_UNLOAD_IDLE_SECS=600`（10 分钟）。
- **冲突/多实现选择（后续演进）**：对“同类多模块”（如多翻译/多检索插件），可逐步引入成功率/耗时统计（结合 `task_manager.task_completed` 的 `runtime` 数据与模块级指标），优先加载历史表现更好的实现，形成“插件策略选择器”。

### 9.9 故障感知与自愈调度（重载/降级/切换）

除了“资源压力”，调度还应把**失败/不稳定**作为一类输入信号：当某类任务持续失败，往往意味着模块/后端实现不稳定或外部依赖异常。建议将故障处理拆成两层：

- **快速自愈（短窗口、低成本）**：重载模块、重试一次、切换到更稳的后端（降级），优先保证关键链路可用性。
- **深度自愈（语义诊断、修复计划）**：交给 `SelfDiagnoser`/`SelfDebugManager` 等产生解释与修复方案（更慢但更彻底）。

#### 9.9.1 故障信号输入

- **任务失败事件**：`task_manager.task_completed` 中 `status!=success` 时的 `error`/`metadata`。
- **动作结果事件**：`agent.action.outcome`（若已在系统内形成统一的 outcome 规范，可做更细的失败分类）。
- **诊断输出事件**：`diagnostics.self_diagnosis` / `planner.plan_ready`（用于触发更高层的修复/回滚/重新规划）。

#### 9.9.2 建议动作（可落地的规则）

1) **模块重载（capability/module reload）**  
当失败任务在 metadata 中明确标注模块（如 `metadata["module"]`/`["capability"]`），且在短窗口内失败次数达到阈值，则对该模块执行一次 unload+load 以清理异常状态。

- 参考实现：`backend/execution/fault_recovery_manager.py::FaultRecoveryManager`（默认关闭，避免误触发）
- 配置示例：
  - `FAULT_RECOVERY_ENABLED=1`
  - `FAULT_RECOVERY_MAX_FAILURES=3`
  - `FAULT_RECOVERY_WINDOW_SECS=180`
  - `FAULT_RECOVERY_COOLDOWN_SECS=600`
- 可观测事件：`fault_recovery.module_reload_attempted` / `fault_recovery.module_reloaded`

2) **脑后端运行时切换（BrainSimulation → WholeBrain）**  
当本地脑后端在运行期抛出异常（例如仿真后端不稳定），先进行一次“重启同后端”尝试；若在窗口内持续失败则切换到更稳的后端以保障可用性。

- 参考实现：`third_party/autogpt/autogpt/core/agent/cognition.py::SimpleBrainAdapter`（默认开启，可按环境变量关闭）
- 配置示例：
  - `BRAIN_RUNTIME_FAILOVER_ENABLED=1`
  - `BRAIN_RUNTIME_RESTART_ON_FAILURE=1`
  - `BRAIN_RUNTIME_FAILOVER_THRESHOLD=3`
  - `BRAIN_RUNTIME_FAILOVER_WINDOW_SECS=180`
  - `BRAIN_RUNTIME_FAILOVER_COOLDOWN_SECS=900`
- 可观测事件：`brain.backend.failure` / `brain.backend.restart` / `brain.backend.failover`

#### 9.9.3 防抖与闭环建议

- **滑动窗口 + 冷却时间**：避免频繁重载/切换导致抖动（thrashing）。
- **只对“明确归因”的失败采取动作**：例如仅当任务携带模块标识时触发模块重载，避免把业务逻辑错误当成模块故障。
- **把故障信号汇入统一观测面**：结合资源事件（`agent.resource`/`resource.signal`）与故障事件（`task_manager.*`/`brain.backend.*`）做综合决策：资源过载时倾向限流，故障突发时倾向重载/降级。

### 9.10 调度控制逻辑的插入位置（集中在控制平面）

为了保持代码结构清晰、避免把调度策略散落在业务模块内部，建议把“策略决策/控制面”集中放在少数几个核心管理点，其他组件只提供**观测数据**与**可调旋钮**。

#### 9.10.1 首选插入点：`AgentLifecycleManager`（全局协调）

`backend/execution/manager.py::AgentLifecycleManager` 已经天然聚合了调度所需的关键组件（`EventBus`、`TaskManager`、`ResourceScheduler`、模块管理器、Agent 列表与状态），因此是最适合承载“调度控制逻辑”的位置：

- **输入**：订阅 `agent.resource`、`task_manager.*`、`environment.*`、`resource.signal` 等事件（或直接读取短窗历史）。
- **状态**：维护全局 backlog、持续负载窗口、故障计数等（避免在每个模块里各自维护一套）。
- **输出/动作**：调用 `TaskManager` 的并发 gate（限流/恢复）、触发模块预热/卸载、必要时发布策略事件（例如 `environment.alert`、`module.lifecycle.*`）驱动其他组件进入保守模式。

工程上建议以“独立控制器组件”方式落地：在 `AgentLifecycleManager` 中只负责 **wiring**（实例化/关闭/注册订阅），具体策略封装成单独的 manager（类似 `ModuleLifecycleManager`、`FaultRecoveryManager`），避免 `manager.py` 继续膨胀。

#### 9.10.2 次级插入点：`TaskManager`（提供旋钮，不写策略）

`backend/execution/task_manager.py::TaskManager` 更适合作为“执行层/执行器”：

- 提供 **并发 limiter**（`set_device_concurrency_limit()`）、**设备路由**（`device="auto"`）与 **扫描派发**（避免头部阻塞）等中性能力。
- 避免把“何时限流/如何降级”的规则硬编码在 TaskManager 内；策略应该由上层（控制平面）根据监控数据决定。
- 任务优先级建议主要在提交时确定；若需要“动态提升/降低优先级”，建议通过“取消并重投递”或后续新增 `reprioritize()` 这类显式接口实现。

#### 9.10.3 辅助插入点：专用策略管理器（单一职责）

为避免“一个超级调度器管一切”，建议把特定策略拆成可插拔的 manager：

- **资源/注意力调度**：`backend/monitoring/resource_scheduler.py::ResourceScheduler`（面向全局 workspace 注意力阈值与模块轮询节奏）
- **模块生命周期**：`backend/execution/module_lifecycle_manager.py::ModuleLifecycleManager`（空闲回收/建议卸载）
- **故障自愈**：`backend/execution/fault_recovery_manager.py::FaultRecoveryManager`（失败突发触发模块重载）
- **脑后端自愈/降级**：`third_party/autogpt/autogpt/core/agent/cognition.py::SimpleBrainAdapter`（运行期失败触发重启/切换）

这些 manager 可以在 `AgentLifecycleManager` 统一挂载，通过事件总线形成闭环：**观测 → 决策 → 动作 → 观测**。

### 9.11 TaskManager 外围调度层：提交包装 + 控制事件

为进一步做到“策略与执行解耦”，建议在 `TaskManager` 之外提供两类能力：

1) **入队前决策（Admission / Submit wrapper）**：集中决定 `priority/device/metadata`，再交由 `TaskManager` 排队  
   - 参考实现：`backend/execution/task_submission_scheduler.py::TaskSubmissionScheduler`
   - 推荐接口形态：`submit_task(func, ..., metadata)`（内部调用 `TaskManager.submit(...)`）

2) **事件驱动控制面（Control plane via EventBus）**：调度器只发布命令，各执行方订阅并执行  
   - 参考实现：`backend/execution/scheduler_control_manager.py::SchedulerControlManager`
   - 核心事件：
     - `scheduler.control`：调度控制指令
     - `scheduler.status`：调度状态/动作回执（可观测）

#### 9.11.1 `scheduler.control`（建议载荷）

节流并发（示例）：

```json
{
  "action": "throttle",
  "device": "cpu",
  "concurrency": 4,
  "reason": "sustained_high_cpu",
  "source": "environment_adapter"
}
```

调整 Agent 选择权重（示例）：

```json
{
  "action": "set_weights",
  "weights": {"cpu": 1.0, "memory": 1.5, "tasks": 0.5},
  "source": "scheduler"
}
```

#### 9.11.2 `scheduler.status`（建议载荷）

控制面动作回执/状态快照（示例）：

```json
{
  "time": 1730000000.0,
  "trigger": "throttle",
  "action": "throttle",
  "queue_depth": 12,
  "device_concurrency_limits": {"cpu": 4}
}
```

落地建议：

- **控制面集中**：由 `AgentLifecycleManager` 负责 wiring（创建/关闭、订阅/发布），策略组件只通过事件交互。
- **执行面简单**：`TaskManager` 保持“队列/执行/并发 gate”职责，不直接写策略；需要新策略时优先新增 wrapper/manager。
- **渐进迁移**：先让新任务入口使用 `TaskSubmissionScheduler`，旧调用点可逐步替换，不要求一次性改完。

### 9.12 模块接口层（Hooks）：配置热更新/降级（非侵入式）

为了避免把“调度判断”散落到各业务模块内部，推荐用 **事件驱动 + 可选接口** 的方式做参数调节与降级：

- 调度器只发布 `scheduler.control` 命令（控制面）。
- 被调度方（TaskManager/脑后端/能力模块）订阅并执行（执行面）。
- 模块不强制改 `ModuleInterface`；只要实现可选的 `update_config(...)` 即可被控制面调用。

目前已落地的 hooks：

- **脑后端配置热更新**：`third_party/autogpt/autogpt/core/agent/cognition.py::SimpleBrainAdapter` 订阅 `scheduler.control`，处理 `brain.update_config` / `brain.switch_backend`（可用 `agent_id` 定向）。
- **BrainSimulation 参数调节**：`modules/brain/backends/__init__.py::BrainSimulationSystemAdapter.update_config` 支持 `dt/timestep_ms` 更新，并在后台连续仿真开启时尝试重启循环以应用新步长。
- **能力模块配置热更新**：`backend/execution/scheduler_control_manager.py::SchedulerControlManager` 支持 `module.update_config`（需要注入 `RuntimeModuleManager`，`AgentLifecycleManager` 已在创建后 attach）。

#### 9.12.1 `brain.update_config`（示例）

```json
{
  "action": "brain.update_config",
  "agent_id": "alpha",
  "overrides": {"simulation": {"dt": 200.0}},
  "source": "scheduler"
}
```

说明：
- `agent_id` 可选；缺省为广播（所有订阅的 brain adapter 都会尝试执行）。
- `overrides`/`runtime_config` 会 best-effort 传给后端的 `update_config`（若后端未实现则忽略）。

#### 9.12.2 `brain.switch_backend`（示例）

```json
{
  "action": "brain.switch_backend",
  "agent_id": "alpha",
  "backend": "WHOLE_BRAIN",
  "reason": "sustained_high_load",
  "source": "scheduler"
}
```

#### 9.12.3 `module.update_config`（示例）

```json
{
  "action": "module.update_config",
  "module": "database",
  "overrides": {"pool_size": 2},
  "load": false,
  "source": "scheduler"
}
```

说明：
- 目标模块是否支持由实现决定；未实现 `update_config` 时会被忽略。
- `load=true` 可在未加载时先加载再更新（谨慎使用，避免错误指令导致意外加载）。

### 9.13 GlobalWorkspace 注意力阈值：全局“削峰/降噪”调度旋钮

`backend/monitoring/global_workspace.py::GlobalWorkspace` 的 `attention_threshold` 可以作为一种 **认知式调度** 的全局旋钮：

- **高负载/队列堆积**：提高阈值 → 低激活（低注意力/低重要性）的广播不会进入“意识层”，从而自动减少跨模块传播与下游处理开销。
- **高严重告警**：降低阈值 → 更多信号可穿透（优先保障响应性）。

落地上，`backend/monitoring/resource_scheduler.py::ResourceScheduler` 已根据 `agent.resource`（CPU/内存）+ backlog + `environment.alert` 计算并调用 `set_attention_threshold()`，形成“观测→调节”的闭环。

#### 9.13.1 模块约定（让阈值真正生效）

为保证这套机制对模块 **非侵入、但有效**，建议建立两条开发约定：

- **为广播附上强度信息**：发布到 workspace 的消息/状态尽量提供 `attention`（或至少填 `importance`），用于阈值过滤。
  - `publish_message(..., propagate=True)` 在未显式提供 `attention` 时，会 best-effort 用 `message.importance`（0~1 clamp）作为默认注意力。
- **在模块内部做轻量早退（可选）**：对“可以延后”的重计算流程，可读取 `global_workspace.attention_threshold()` 做粗粒度早退或降低频率（例如：阈值越高 → 轮询越慢、采样更稀疏）。

这种方式的优势是：调度层只调一个“阈值”，各模块只需遵守统一约定即可自然降噪/限流，不需要把复杂策略散落到各处。

### 9.14 解耦、兼容与开关（渐进增强）

调度应作为“增强层”渐进引入：即使调度组件失效/关闭，也应退化到现有行为而不破坏正确性。

#### 9.14.1 解耦原则（现有落地点）

- **策略与执行分离**：策略通过 `scheduler.control` 下发，执行方各自订阅并落地（`SchedulerControlManager` / `SimpleBrainAdapter` / `TaskManager` 等）。
- **可选接口而非强制改造**：模块只需按需实现 `update_config(...)`，不要求改 `ModuleInterface` 统一增加新方法。
- **集中 wiring**：在 `AgentLifecycleManager` 统一创建/关闭调度相关组件，业务模块避免直接感知调度策略。

#### 9.14.2 兼容退化（Failure-safe）

- `TaskSubmissionScheduler` 关闭：任务提交按原始参数进入 `TaskManager`（仍可执行，只是少了入队前启发式与状态事件）。
- `SchedulerControlManager` 关闭：`scheduler.control` 事件将被忽略；`EnvironmentAdapter` 的并发调节会回退为直接调用 `TaskManager.set_device_concurrency_limit()`。
- 脑后端切换失败：`SimpleBrainAdapter` 保持当前后端继续运行（同时会有 `brain.backend.failover` / `brain.backend.restart` 等事件记录失败）。
- 模块卸载/回收未及时发生：只影响资源占用，不影响主流程执行正确性。

#### 9.14.3 配置开关（建议/已提供）

可通过环境变量按需启用局部能力，便于灰度与排障：

- `TASK_SUBMISSION_SCHEDULER_ENABLED`：是否启用入队前决策包装。
- `SCHEDULER_CONTROL_ENABLED`：是否启用 `scheduler.control` 控制面执行器。
- `SCHEDULER_STATUS_ON_SUBMIT`：是否在提交时发 `scheduler.status`（可观测性）。
- `BRAIN_SCHEDULER_CONTROL_ENABLED`：是否允许 brain adapter 响应 `scheduler.control`（热更新/切换后端）。
- `BRAIN_RUNTIME_FAILOVER_ENABLED` / `BRAIN_RUNTIME_RESTART_ON_FAILURE`：脑后端运行期自愈/降级开关。
- `TASK_MANAGER_DYNAMIC_CONCURRENCY`：是否允许 `EnvironmentAdapter` 动态调节并发（高负载时限流）。

#### 9.14.4 可观测性（事件建议）

建议以事件为“自解释日志”：

- `scheduler.status`：控制面动作回执（并发调整、权重调整、模块/脑配置更新等）。
- `task_manager.*`：任务开始/完成/失败与 runtime 指标。
- `module.loaded/module.used/module.unloaded`：模块生命周期与命中情况。
- `brain.backend.failure/restart/failover`：脑后端故障与切换轨迹。
- `monitoring.scheduler`（workspace message）：阈值/负载/backlog/告警与模块轮询间隔快照。
