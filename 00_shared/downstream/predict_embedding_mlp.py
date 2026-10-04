from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common import EmbeddingMLP, load_pair


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply a common embedding MLP checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--embedding", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    values, metadata = load_pair(args.embedding, args.metadata)
    if values.shape[1] != int(checkpoint["input_dim"]):
        raise ValueError(
            f"Embedding dimension {values.shape[1]} does not match checkpoint "
            f"dimension {checkpoint['input_dim']}"
        )
    mean = checkpoint["embedding_mean"].cpu().numpy()
    std = checkpoint["embedding_std"].cpu().numpy()
    values = ((values - mean) / std).astype(np.float32)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    model = EmbeddingMLP(
        int(checkpoint["input_dim"]),
        checkpoint["hidden_dims"],
        float(checkpoint["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(values), args.batch_size):
            batch = torch.from_numpy(values[start : start + args.batch_size]).to(device)
            outputs.append(model(batch).cpu().numpy())
    raw = np.concatenate(outputs)
    result = metadata.copy()
    if checkpoint["task"] == "classification":
        probability = torch.sigmoid(torch.from_numpy(raw)).numpy()
        threshold = float(checkpoint["threshold"])
        result["senescent_probability"] = probability
        result["predicted_label"] = (probability >= threshold).astype(np.int8)
    else:
        result["predicted_age"] = raw
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Saved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
