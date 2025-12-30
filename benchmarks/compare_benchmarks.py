import json
import sys
from pathlib import Path


def main(baseline_path: str, new_path: str, threshold: float = 0.1) -> None:
    baseline = json.loads(Path(baseline_path).read_text())
    new = json.loads(Path(new_path).read_text())

    base_map = {b["name"]: b["stats"]["median"] for b in baseline.get("benchmarks", [])}
    regressions = []
    for bench in new.get("benchmarks", []):
        name = bench["name"]
        if name in base_map:
            base_median = base_map[name]
            new_median = bench["stats"]["median"]
            if new_median > base_median * (1 + threshold):
                regressions.append((name, base_median, new_median))

    if regressions:
        print("Performance regression detected:")
        for name, base_median, new_median in regressions:
            print(f"{name}: baseline {base_median:.6f}s, current {new_median:.6f}s")
        sys.exit(1)
    else:
        print("No performance regression detected.")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: compare_benchmarks.py <baseline> <new> [threshold]")
        sys.exit(2)
    baseline = sys.argv[1]
    new = sys.argv[2]
    thr = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1
    main(baseline, new, thr)
