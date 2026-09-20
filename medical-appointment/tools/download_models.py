"""Fetch model weights into /mnt/data/nordicai/medical-appointment/models.

    python tools/download_models.py                 # serving set: Qwen3-ASR-1.7B, aligner, whisper turbo, Qwen3.5-9B
    python tools/download_models.py --set smoke     # small CPU set for smoke tests
    python tools/download_models.py --set extra     # alternatives worth benchmarking
"""

import argparse
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from settings import MODELS  # noqa: E402

SNAPSHOTS = {
    'default': [
        'Qwen/Qwen3-ASR-1.7B-hf',
        'Qwen/Qwen3-ForcedAligner-0.6B-hf',
        'mobiuslabsgmbh/faster-whisper-large-v3-turbo',
    ],
    'smoke': [
        'Qwen/Qwen3-ASR-0.6B-hf',
        'Qwen/Qwen3-ForcedAligner-0.6B-hf',
    ],
    'extra': [
        'Systran/faster-whisper-large-v3',
    ],
}

GGUF = {
    'default': [('unsloth/Qwen3.5-9B-GGUF', 'Qwen3.5-9B-Q4_K_M.gguf')],
    'smoke': [('bartowski/Qwen_Qwen3.5-0.8B-GGUF', 'Qwen_Qwen3.5-0.8B-Q4_K_M.gguf')],
    'extra': [
        ('unsloth/gemma-4-12B-it-qat-GGUF', 'gemma-4-12B-it-qat-UD-Q4_K_XL.gguf'),
        ('unsloth/Qwen3.5-4B-GGUF', 'Qwen3.5-4B-Q4_K_M.gguf'),
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--set', choices=sorted(SNAPSHOTS), default='default')
    args = parser.parse_args()

    for repo in SNAPSHOTS[args.set]:
        target = MODELS / repo.split('/')[-1]
        print(f'{repo} -> {target}', flush=True)
        snapshot_download(repo, local_dir=target)

    for repo, filename in GGUF[args.set]:
        target = MODELS / 'gguf'
        print(f'{repo}/{filename} -> {target}', flush=True)
        hf_hub_download(repo, filename, local_dir=target)

    print('done', flush=True)


if __name__ == '__main__':
    main()
