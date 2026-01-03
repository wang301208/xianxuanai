# Brain 后端选择

默认认知后端为 `brain_simulation`（BrainSimulationSystem 适配器）。

## 覆盖方式（从易到细）

- 全局环境变量：`BRAIN_BACKEND=brain_simulation|whole_brain|transformer|llm`
- CLI 参数：`--brain-backend ...`（等价于设置 `BRAIN_BACKEND`）
- Agent Blueprint：在 blueprint 中设置 `brain_backend: ...`（会覆盖全局配置）

## 备注

- `transformer` 需要 PyTorch；缺失依赖时会自动回退到 `llm`。
- `whole_brain` / `brain_simulation` 初始化失败时也会回退到 `llm`。
- BrainSimulationSystem 的 profile/stage 可用 `BRAIN_SIM_PROFILE`、`BRAIN_SIM_STAGE` 控制（默认 `BRAIN_SIM_PROFILE=prototype`）。
