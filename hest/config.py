"""Repository paths shared by the scripts: dataset files, the default output directory, path resolution."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = {"math500": os.path.join(ROOT, "data", "math500_test.json"),
        "gsm8k": os.path.join(ROOT, "data", "gsm8k_test.json"),
        "mathtrain": os.path.join(ROOT, "data", "math_train2000.json"),
        "gsmtrain": os.path.join(ROOT, "data", "gsm8k_train.json"),
        "olympiad": os.path.join(ROOT, "data", "olympiadbench_test.json")}
OUT_DIR = os.path.join(ROOT, "outputs")
CHECKPOINTS = os.path.join(ROOT, "data", "checkpoints")


def resolve(p):
    """a path given relative to the repository root, or absolute"""
    return p if os.path.isabs(p) else os.path.join(ROOT, p)


def out_path(*parts):
    """a path under outputs/ (directories created)"""
    p = os.path.join(OUT_DIR, *parts)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p
