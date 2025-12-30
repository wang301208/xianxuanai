from BrainSimulationSystem.config.stage_profiles import build_stage_config


def test_infant_stage_scales_brain_regions_down():
    cfg = build_stage_config("infant")

    assert cfg["metadata"]["stage"] == "infant"
    total_neurons = cfg["scope"]["total_neurons"]
    full_total = build_stage_config("full")["scope"]["total_neurons"]
    assert total_neurons < full_total * 0.001

    prefrontal = cfg["brain_regions"]["prefrontal_cortex"]
    assert prefrontal["volume"] < 1_000
    assert prefrontal["neuron_density"] < 15_000


def test_stage_overrides_have_priority():
    cfg = build_stage_config("infant", overrides={"scope": {"total_neurons": 123}})
    assert cfg["scope"]["total_neurons"] == 123


def test_stage_config_overrides_enable_modules_and_scale_scope():
    infant = build_stage_config("infant")
    assert infant["scope"]["columns_per_region"] == 1
    assert infant["scope"]["column_total_neurons"] == 64
    assert infant["perception"]["vision"]["enabled"] is True
    assert infant["perception"]["vision"]["model"]["backend"] == "numpy"
    assert infant["perception"]["vision"]["model"]["input_size"] == (64, 64)
    assert infant["perception"]["auditory"]["enabled"] is True
    assert infant["perception"]["auditory"]["model"]["backend"] == "numpy"
    assert infant["memory"]["system"]["working_memory"]["capacity"] == 7
    assert infant["memory"]["system"]["working_memory"]["strategy"] == "priority"
    assert infant["memory"]["experience"]["enabled"] is True
    assert infant["memory"]["experience"]["interval_steps"] == 2
    assert infant["learning"]["interactive_language_loop"]["mentor_interval"] == 1
    assert infant["learning"]["mentor"]["enabled"] is True
    assert infant["learning"]["offline_training"]["enabled"] is True
    assert infant["environment"]["kind"] == "toy_room"

    modules = infant.get("modules") or {}
    assert "language_hub" not in (modules.get("components") or {})

    juvenile = build_stage_config("juvenile")
    assert juvenile["scope"]["columns_per_region"] == 2
    assert juvenile["perception"]["multimodal_fusion"]["enabled"] is True
    assert juvenile["memory"]["system"]["working_memory"]["capacity"] == 16
    assert juvenile["memory"]["system"]["working_memory"]["strategy"] == "indexed"
    assert juvenile["learning"]["interactive_language_loop"]["mentor_interval"] == 4
    assert juvenile["learning"]["reward_shaping"]["success_bonus"] > 0
    assert juvenile["environment"]["kind"] == "toy_teacher"
    juvenile_components = (juvenile.get("modules") or {}).get("components") or {}
    assert "language_hub" in juvenile_components
    assert juvenile_components["language_hub"]["llm_service"]["enabled"] is False

    adolescent = build_stage_config("adolescent")
    assert adolescent["scope"]["columns_per_region"] == 3
    assert adolescent["scope"]["column_total_neurons"] == 128
    assert adolescent["metacognition"]["enabled"] is True
    assert adolescent["meta_reasoning"]["enabled"] is True
    assert adolescent["memory"]["system"]["working_memory"]["capacity"] == 32
    assert adolescent["learning"]["interactive_language_loop"]["mentor_interval"] >= 4
    assert adolescent["learning"]["offline_training"]["batch_size"] >= 16
    assert adolescent["environment"]["kind"] == "open_world"
    adolescent_components = adolescent["modules"]["components"]
    assert adolescent_components["language_hub"]["llm_service"]["provider"] == "internal_pipeline"
