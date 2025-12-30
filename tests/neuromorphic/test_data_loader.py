import json
from pathlib import Path

def test_dataset_loader(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "config.json").write_text('{"n_neurons":1}', encoding="utf-8")
    (dataset / "signal.json").write_text('[[1],[0]]', encoding="utf-8")
    (dataset / "target.json").write_text('[[0],[0]]', encoding="utf-8")

    from modules.brain.neuromorphic.data import DatasetLoader

    loader = DatasetLoader(dataset)
    assert Path(loader.load_config()).name == "config.json"
    assert loader.read_json("signal.json") == [[1.0], [0.0]]
