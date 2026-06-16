#!/usr/bin/env python3
"""Upload the trained RNA FM checkpoints to the Hugging Face Hub.

Creates one model repo (default <user>/RNA_FM) and uploads each model under its
own subfolder. Only the final model + tokenizer + config files are uploaded
(checkpoint-*/ and optimizer states are skipped).

Run:
    python scripts/upload_to_hf.py --token hf_xxx --username YOURNAME
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi, create_repo, upload_folder

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# subfolder -> local model dir (real paths, not symlinks)
MODELS = {
    "kmer1-BERT":      "outputs/bert_mlm/fm_single",
    "kmer1-Hybrid":    "outputs/fm_hybrid_mamba_kmer1",
    "kmer1-decoder":   "outputs/fm_decoder_kmer1",
    "kmer6-BERT":      "outputs/bert_mlm/fm_kmer6",
    "kmer6-Hybrid":    "outputs/fm_hybrid_mamba_kmer6",
    "kmer6-decoder":   "outputs/fm_decoder_kmer6",
    "kmer1-Hybrid-300M": "outputs/fm_hybrid_mamba_kmer1_300m",
    # Hybrid-650M, epoch-1 checkpoint (RiNALMo-Giga-aligned run; pretraining to be
    # finished on H200 — see HANDOFF.md). Clean LM weights, no optimizer state.
    "kmer1-Hybrid-650M-ep1": "outputs/fm_hybrid_650m/ss_ft_ep1_model",
}
# only top-level final-model files (bare names -> won't match checkpoint-*/ subdirs)
ALLOW = ["config.json", "model.safetensors", "pytorch_model.bin",
         "model_config.json", "tokenizer_config.json", "vocab.txt",
         "special_tokens_map.json", "tokenizer.json", "generation_config.json",
         "eval_results.json"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--token", required=True, help="HF write token")
    p.add_argument("--username", required=True)
    p.add_argument("--repo", default="RNA_FM")
    p.add_argument("--private", action="store_true")
    p.add_argument("--only", nargs="*", help="upload only these subfolders")
    args = p.parse_args()

    repo_id = f"{args.username}/{args.repo}"
    who = HfApi().whoami(token=args.token).get("name")
    print(f"[hf] token account: {who}  -> repo {repo_id} (private={args.private})")
    create_repo(repo_id, token=args.token, private=args.private,
                repo_type="model", exist_ok=True)

    items = MODELS.items() if not args.only else [(k, MODELS[k]) for k in args.only]
    for sub, rel in items:
        path = PROJECT_ROOT / rel
        if not path.exists():
            print(f"[hf] SKIP {sub}: {path} missing")
            continue
        print(f"[hf] uploading {sub}  ({rel}) ...")
        upload_folder(repo_id=repo_id, folder_path=str(path), path_in_repo=sub,
                      token=args.token, allow_patterns=ALLOW,
                      commit_message=f"add {sub}")
        print(f"[hf]   done -> https://huggingface.co/{repo_id}/tree/main/{sub}")

    print(f"\n[hf] ALL DONE -> https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
