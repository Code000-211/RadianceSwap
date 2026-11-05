"""Turn a latent-sparse-attention map into the tensors a Radiance container carries for LSA (docs/LSA.md).

A map file is a `torch.save` dict of `"<layer>.W_Q_idx"` and `"<layer>.W_K_idx"`, each `[key_dim, index_dim]`
fp32 - the release `d-lsa-weights-v1` holds Qwen3.8-27B's. Radiance reads a map as a projection's `[N][K]`,
so each is written transposed, in bf16 (the precision the reference serves it at), under the layer's name:

    model.language_model.layers.<L>.self_attn.lsa.q_map.weight   [index_dim, key_dim] bf16
    model.language_model.layers.<L>.self_attn.lsa.k_map.weight

Only torch is needed, and the safetensors file is written by hand. The command to merge it prints at the end:

    python tools/lsa/export_map.py qwen38_27b__long-16k-1M-heavy-key-and-query-d2048.pt lsa_map.safetensors
    rad-convert <checkpoint> --reuse <container>.rad --in-place --add lsa_map.safetensors \\
        --set lsa.enable=1 ...

`--set` writes the design's knobs into the container's metadata, where the qwen35 plugins read them; the
defaults are the served D-LSA (Flat) design and need not be set.
"""

import argparse
import json
import struct
from pathlib import Path

import torch

# The served design (latent-sparse-attention settings/design.py), the plugin's defaults
DESIGN = {
    "lsa.index_dim": 64,
    "lsa.pool_size": 4,
    "lsa.cluster_block": 512,
    "lsa.bucket_size": 8,
    "lsa.bucket_cap": 16,
    "lsa.window_size": 512,
    "lsa.sink_size": 4,
    "lsa.top_p": 1024,
    "lsa.top_k": 4096,
}


def name(layer: int, which: str) -> str:
    return f"model.language_model.layers.{layer}.self_attn.lsa.{which}_map.weight"


def write_safetensors(path: Path, tensors: dict[str, torch.Tensor], metadata: dict[str, str]) -> None:
    header: dict[str, object] = {"__metadata__": metadata}
    offset = 0
    blobs = []
    for key in sorted(tensors):
        data = tensors[key].contiguous().view(torch.uint16).numpy().tobytes()
        header[key] = {"dtype": "BF16", "shape": list(tensors[key].shape), "data_offsets": [offset, offset + len(data)]}
        offset += len(data)
        blobs.append(data)
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    with open(path, "wb") as out:
        out.write(struct.pack("<Q", len(encoded)))
        out.write(encoded)
        for blob in blobs:
            out.write(blob)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("map", type=Path, help="a latent-sparse-attention map (.pt)")
    parser.add_argument("out", type=Path, help="the safetensors file to write")
    parser.add_argument("--key-dim", type=int, default=None, help="refuse a map whose input is not this wide")
    args = parser.parse_args()

    maps = torch.load(args.map, map_location="cpu", weights_only=True)
    layers = sorted({int(key.split(".")[0]) for key in maps})
    tensors = {}
    for layer in layers:
        for which, key in (("q", "W_Q_idx"), ("k", "W_K_idx")):
            weight = maps.get(f"{layer}.{key}")
            if weight is None or weight.ndim != 2:
                raise SystemExit(f"layer {layer} has no {key} of rank 2")
            if args.key_dim is not None and weight.shape[0] != args.key_dim:
                raise SystemExit(f"layer {layer}'s {key} reads {weight.shape[0]} dims, not {args.key_dim}")
            tensors[name(layer, which)] = weight.T.to(torch.bfloat16)
    index_dim = {int(t.shape[0]) for t in tensors.values()}
    if index_dim != {DESIGN["lsa.index_dim"]}:
        raise SystemExit(f"the map is {sorted(index_dim)} dims wide, the design {DESIGN['lsa.index_dim']}")
    write_safetensors(args.out, tensors, {"source": args.map.name, "layers": ",".join(map(str, layers))})

    print(f"{args.out}: {len(layers)} layers {layers}, maps {list(next(iter(tensors.values())).shape)} bf16")
    sets = " ".join(f"--set {key}={value}" for key, value in [("lsa.enable", 1), *DESIGN.items()])
    print(f"rad-convert <checkpoint> --reuse <container>.rad --in-place --add {args.out} {sets}")


if __name__ == "__main__":
    main()
