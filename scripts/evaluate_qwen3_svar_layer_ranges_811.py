"""Evaluate Qwen3 native SVAR on decoder layer slices [5,20) and [5,21)."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import evaluate_qwen25_svar_layer_ranges_811 as experiment


experiment.MODEL = "qwen3_vl_8b"
experiment.RANGES = ((5, 20), (5, 21))
experiment.OUT = experiment.ROOT / "outputs/qwen3_svar_layer_ranges_811_v1"
experiment.SCHEMA = "qwen3-native-svar-layer-ranges-811-v1"


if __name__ == "__main__":
    experiment.main()
