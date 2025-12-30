import json
import sys
from pathlib import Path
import types

import pandas as pd

# Create a lightweight stub for torch to avoid heavy dependency during tests
sys.modules.setdefault("torch", types.SimpleNamespace())
pil_module = types.ModuleType("PIL")
pil_image = types.ModuleType("Image")
pil_module.Image = pil_image
sys.modules.setdefault("PIL", pil_module)
sys.modules.setdefault("PIL.Image", pil_image)
sys.path.append(str(Path(__file__).resolve().parent.parent))


def _create_dataset(path: Path) -> None:
    data = {
        "text": ["aa", "bb", "aa bb", "bb cc", "aa cc", "cc dd"],
        "target": [0, 1, 0, 1, 0, 1],
    }
    df = pd.DataFrame(data)
    df.to_csv(path, index=False)


def test_hyperparameter_search(tmp_path: Path, capsys) -> None:
    csv_path = tmp_path / "data.csv"
    _create_dataset(csv_path)

    version = "test_run"
    args = [
        "train_models.py",
        str(csv_path),
        "--model",
        "linear",
        "--cv",
        "2",
        "--search-space",
        json.dumps({"fit_intercept": [True, False]}),
        "--version",
        version,
    ]

    from backend.ml import train_models

    sys.argv = args
    train_models.main()
    captured = capsys.readouterr()
    assert "Cross-validation MSE" in captured.out

    metrics_file = Path("artifacts") / version / "metrics.txt"
    with open(metrics_file) as f:
        contents = f.read()
    assert "Best Params" in contents and "fit_intercept" in contents

    # cleanup
    for p in metrics_file.parent.glob("*"):
        p.unlink()
    metrics_file.parent.rmdir()
