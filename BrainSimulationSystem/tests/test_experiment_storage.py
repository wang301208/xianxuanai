import asyncio
import json

from BrainSimulationSystem.infrastructure.experiment_storage import ExperimentStorage
from BrainSimulationSystem.infrastructure.simulation_pipeline import (
    ExperimentConfig,
    OptimizationMethod,
    ParameterSpace,
    SimulationPipeline,
)


def test_experiment_storage_initializes_layout(tmp_path) -> None:
    storage = ExperimentStorage(tmp_path, "exp_test", name="demo", description="desc")
    storage.initialize()

    assert storage.dirs.root.is_dir()
    assert storage.dirs.root.parent == tmp_path
    assert storage.dirs.root.name.startswith("experiment_exp_test_")
    assert storage.dirs.config.is_dir()
    assert storage.dirs.data.is_dir()
    assert storage.dirs.results.is_dir()
    assert storage.dirs.model.is_dir()
    assert storage.dirs.sim.is_dir()
    assert storage.dirs.inputs.is_dir()
    assert storage.dirs.outputs.is_dir()
    assert storage.dirs.checkpoints.is_dir()
    assert storage.dirs.logs.is_dir()

    meta = json.loads((storage.dirs.root / "meta.json").read_text(encoding="utf-8"))
    assert meta["schema_version"] == "exp-meta-v1"
    assert meta["experiment_id"] == "exp_test"
    assert meta["name"] == "demo"
    assert meta["project"]["name"] == "BrainSimulationSystem"
    assert "responsible" in meta
    assert "tags" in meta
    assert meta["storage"]["layout"] == "config-data-results-v1"

    manifest = json.loads((storage.dirs.root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "exp-manifest-v1"
    assert manifest["experiment_id"] == "exp_test"
    assert manifest["artifacts"] == []

    assert (storage.dirs.root / "provenance.json").exists()


def test_experiment_storage_registers_artifacts(tmp_path) -> None:
    storage = ExperimentStorage(tmp_path, "exp_test")
    storage.initialize()

    storage.write_text("results/hello.txt", "hello", kind="log")
    storage.write_jsonl_gz(
        "config/inputs/datasets/transitions.jsonl.gz",
        [{"step": 0, "reward": 1.0}, {"step": 1, "reward": 0.5}],
        kind="training_dataset",
    )
    storage.write_timeseries_npz(
        "results/training/training_curves.npz",
        {"loss": [1.0, 0.5, 0.25], "reward": [0.1, 0.2, 0.3]},
        kind="training_log",
    )
    manifest = json.loads((storage.dirs.root / "manifest.json").read_text(encoding="utf-8"))
    assert any(a["path"] == "results/hello.txt" and a["kind"] == "log" for a in manifest["artifacts"])
    assert any(
        a["path"] == "config/inputs/datasets/transitions.jsonl.gz" and a["kind"] == "training_dataset"
        for a in manifest["artifacts"]
    )
    assert any(
        a["path"] == "results/training/training_curves.npz" and a["kind"] == "training_log"
        for a in manifest["artifacts"]
    )

    storage.write_text("results/hello.txt", "hello world", kind="log")
    manifest2 = json.loads((storage.dirs.root / "manifest.json").read_text(encoding="utf-8"))
    artifacts = [a for a in manifest2["artifacts"] if a["path"] == "results/hello.txt"]
    assert len(artifacts) == 1
    assert artifacts[0]["bytes"] == len("hello world".encode("utf-8"))


def test_simulation_pipeline_writes_experiment_layout(tmp_path) -> None:
    config = ExperimentConfig(
        experiment_id="exp_test",
        name="pipeline-test",
        description="desc",
        parameter_spaces=[
            ParameterSpace(
                name="spike_rate",
                param_type="continuous",
                bounds=(5.0, 6.0),
            )
        ],
        optimization_method=OptimizationMethod.RANDOM_SEARCH,
        optimization_config={"num_samples": 2},
        base_network_config={"num_neurons": 10},
        base_simulation_params={"simulation_time": 10.0},
        responsible="tester",
        tags=["unit-test"],
        metadata={"team": "qa"},
        project_version="9.9.9",
        max_parallel_jobs=2,
        output_directory=str(tmp_path),
        experiment_dirname="experiment_exp_test",
        save_intermediate_results=True,
        random_seed=123,
    )

    pipeline = SimulationPipeline(config)
    result = asyncio.run(pipeline.run_experiment())
    assert result.experiment_id == "exp_test"

    experiment_dir = tmp_path / "experiment_exp_test"
    assert (experiment_dir / "meta.json").exists()
    assert (experiment_dir / "manifest.json").exists()
    assert (experiment_dir / "config" / "inputs" / "input_spec.json").exists()
    assert (experiment_dir / "config" / "sim" / "seeds.json").exists()
    assert (experiment_dir / "config" / "config.json").exists()
    assert (experiment_dir / "data" / "network_structure.json").exists()
    assert (experiment_dir / "results" / "training_log.csv").exists()
    assert (experiment_dir / "results" / "experiment_result.json").exists()
    assert (experiment_dir / "results" / "logs" / "pipeline.log").exists()

    detailed_results_file = experiment_dir / "results" / "detailed_results.h5"
    if not detailed_results_file.exists():
        detailed_results_file = experiment_dir / "results" / "detailed_results.json"
    assert detailed_results_file.exists()
    assert (experiment_dir / "results" / "results_summary.json").exists()

    spikes_file = experiment_dir / "data" / "spikes.h5"
    if not spikes_file.exists():
        spikes_file = experiment_dir / "data" / "spikes.npz"
    assert spikes_file.exists()

    states_file = experiment_dir / "data" / "states.h5"
    if not states_file.exists():
        states_file = experiment_dir / "data" / "states.npz"
    assert states_file.exists()

    job_files = list((experiment_dir / "results" / "jobs").glob("*.json"))
    assert len(job_files) == 2
    job_ids = {path.stem for path in job_files}

    for job_id in job_ids:
        assert (experiment_dir / "config" / "sim" / "jobs" / f"{job_id}.json").exists()
        assert (experiment_dir / "data" / "checkpoints" / "initial_states" / f"{job_id}.json").exists()
        assert (experiment_dir / "data" / "checkpoints" / "initial_states" / f"{job_id}.npz").exists()

    meta = json.loads((experiment_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["status"] == "completed"
    assert meta["responsible"]["name"] == "tester"
    assert meta["project"]["version"] == "9.9.9"
    assert "unit-test" in meta["tags"]
    assert meta["metadata"]["team"] == "qa"

    manifest = json.loads((experiment_dir / "manifest.json").read_text(encoding="utf-8"))
    artifact_paths = {a["path"] for a in manifest["artifacts"]}
    assert "config/config.json" in artifact_paths
    assert "data/network_structure.json" in artifact_paths
    assert "results/training_log.csv" in artifact_paths
    assert "results/results_summary.json" in artifact_paths
    assert "results/logs/pipeline.log" in artifact_paths
    assert "config/sim/experiment_config.json" in artifact_paths
    assert "config/model/base_network_config.json" in artifact_paths
    assert "config/sim/base_simulation_params.json" in artifact_paths
    assert "config/inputs/input_spec.json" in artifact_paths
    assert "config/sim/seeds.json" in artifact_paths
    assert "results/experiment_result.json" in artifact_paths
    assert any(
        p in {"results/detailed_results.h5", "results/detailed_results.json"} for p in artifact_paths
    )
    assert any(p.startswith("results/jobs/") for p in artifact_paths)
    assert any(p.startswith("config/sim/jobs/") for p in artifact_paths)
    assert any(p.startswith("data/checkpoints/initial_states/") for p in artifact_paths)
    assert any(p in {"data/spikes.h5", "data/spikes.npz"} for p in artifact_paths)
    assert any(p in {"data/states.h5", "data/states.npz"} for p in artifact_paths)

    catalog = (tmp_path / "catalog.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert catalog
    assert any(json.loads(line)["experiment_id"] == "exp_test" for line in catalog)


def test_experiment_storage_write_yaml_optional(tmp_path) -> None:
    storage = ExperimentStorage(tmp_path, "exp_test")
    storage.initialize()

    try:
        storage.write_yaml("config.yaml", {"hello": "world"}, kind="config")
    except ImportError:
        return

    assert (storage.dirs.root / "config.yaml").exists()
    manifest = json.loads((storage.dirs.root / "manifest.json").read_text(encoding="utf-8"))
    assert any(a["path"] == "config.yaml" and a["format"] == "yaml" for a in manifest["artifacts"])


def test_experiment_storage_write_pytables_optional(tmp_path) -> None:
    storage = ExperimentStorage(tmp_path, "exp_test")
    storage.initialize()

    try:
        storage.write_pytables_table(
            "results/training_log.h5",
            [{"step": 0, "loss": 1.0}],
            kind="training_log",
            fieldnames=["step", "loss"],
            table_name="training_log",
        )
    except ImportError:
        return

    assert (storage.dirs.root / "results" / "training_log.h5").exists()
    manifest = json.loads((storage.dirs.root / "manifest.json").read_text(encoding="utf-8"))
    assert any(
        a["path"] == "results/training_log.h5" and a["format"] == "hdf5+pytables"
        for a in manifest["artifacts"]
    )


def test_experiment_storage_sonata_stub_export(tmp_path) -> None:
    storage = ExperimentStorage(tmp_path, "exp_test")
    storage.initialize()

    exported = storage.export_network_to_sonata_stub(
        "data/sonata",
        network_config={"num_neurons": 3, "threshold": -50.0},
        metadata={"note": "unit-test"},
    )
    assert exported["sonata_dir"] == "data/sonata"
    assert exported["nodes"] == "data/sonata/nodes.csv"
    assert (storage.dirs.root / "data" / "sonata" / "nodes.csv").exists()
    assert (storage.dirs.root / "data" / "sonata" / "sonata_config.json").exists()

    manifest = json.loads((storage.dirs.root / "manifest.json").read_text(encoding="utf-8"))
    assert any(
        a["path"] == "data/sonata/nodes.csv" and a["kind"] == "sonata_nodes"
        for a in manifest["artifacts"]
    )
