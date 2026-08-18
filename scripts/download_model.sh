#!/usr/bin/env bash
# Download sarvam-30b. Run this on the GPU node, not the laptop.
#
# 128.6 GB across 26 safetensors shards (weights are stored fp32 on the Hub).
# Budget ~1.5 h at 250 MB/s and provision 500 GB-1 TB of NVMe: the working set is the fp32
# source plus a bf16 copy plus ~20 GB per INT4 checkpoint.
#
#   ./scripts/download_model.sh [target_dir]

set -euo pipefail

MODEL_ID="sarvamai/sarvam-30b"
TARGET="${1:-./models/sarvam-30b}"

command -v hf >/dev/null 2>&1 || {
    echo "error: huggingface CLI not found. Install with: uv pip install huggingface-hub[cli]" >&2
    exit 1
}

echo "==> checking free space"
AVAIL_GB=$(df -Pk "$(dirname "$TARGET")" | awk 'NR==2 {print int($4/1048576)}')
echo "    ${AVAIL_GB} GB available at $(dirname "$TARGET")"
if [ "$AVAIL_GB" -lt 400 ]; then
    echo "warning: <400 GB free. The full working set (fp32 source + bf16 + INT4" >&2
    echo "         checkpoints) needs ~330 GB; 500 GB+ is recommended." >&2
fi

echo "==> downloading ${MODEL_ID} (128.6 GB) to ${TARGET}"
# hf_transfer gives a large speedup on fast links; harmless if unavailable.
HF_HUB_ENABLE_HF_TRANSFER=1 hf download "$MODEL_ID" --local-dir "$TARGET"

echo "==> verifying architecture against configs/model/sarvam-30b.yaml"
python3 - "$TARGET" <<'PY'
import json, sys, pathlib
cfg = json.loads((pathlib.Path(sys.argv[1]) / "config.json").read_text())
expected = {
    "num_hidden_layers": 19,
    "first_k_dense_replace": 1,
    "num_experts": 128,
    "num_experts_per_tok": 6,
    "num_shared_experts": 1,
    "hidden_size": 4096,
    "vocab_size": 262144,
}
bad = {k: (cfg.get(k), v) for k, v in expected.items() if cfg.get(k) != v}
if bad:
    print("MISMATCH between the Hub config and configs/model/sarvam-30b.yaml:")
    for k, (got, want) in bad.items():
        print(f"  {k}: hub={got} expected={want}")
    print("\nThe hook design and parameter budget assume these values. Re-verify")
    print("ARCHITECTURE.md §1.2 before quantizing.")
    sys.exit(1)
print("architecture matches the verified values")
PY

echo "==> done: ${TARGET}"
