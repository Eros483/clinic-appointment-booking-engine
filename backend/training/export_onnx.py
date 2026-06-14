# ----- Export trained ECAPA-TDNN to ONNX @ backend/training/export_onnx.py -----

import argparse
from pathlib import Path

import numpy as np
import torch

from backend.training.train_ecapa import ECAPA_TDNN, N_MELS, C, NUM_CLASSES

CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"
ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"


def export(checkpoint_path: str | None = None, output_path: str | None = None):
    chk_path = (
        Path(checkpoint_path) if checkpoint_path else CHECKPOINT_DIR / "best_model.pt"
    )
    out_path = (
        Path(output_path) if output_path else ARTIFACTS_DIR / "lang_id_ecapa_ta.onnx"
    )

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    model = ECAPA_TDNN(channels=C, num_classes=NUM_CLASSES)
    checkpoint = torch.load(chk_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    dummy_input = torch.randn(1, N_MELS, 100)

    torch.onnx.export(
        model,
        dummy_input,
        str(out_path),
        input_names=["mel"],
        output_names=["logits"],
        dynamic_axes={
            "mel": {0: "batch", 2: "time"},
            "logits": {0: "batch"},
        },
        opset_version=18,
    )

    import onnxruntime as ort

    session = ort.InferenceSession(str(out_path))
    ort_output = session.run(None, {"mel": dummy_input.numpy()})[0]
    pt_output = model(dummy_input).detach().numpy()
    max_diff = np.max(np.abs(ort_output - pt_output))
    print(f"ONNX export validated — max numerical diff: {max_diff:.2e}")
    print(f"Model saved to: {out_path}")

    if max_diff > 1e-4:
        raise RuntimeError(
            f"ONNX validation failed: max diff {max_diff:.2e} exceeds 1e-4"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    export(args.checkpoint, args.output)
