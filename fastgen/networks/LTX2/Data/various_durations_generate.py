"""
Batch video generation from prompts.txt using LTX-2 TI2VidTwoStagesPipeline.
Generates each prompt at 3 durations: 5s, 8s, 10s with the same seed.
Output files: {stem}_5s.mp4, {stem}_8s.mp4, {stem}_10s.mp4
"""

import json
import logging
import sys
from pathlib import Path

import torch

from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_pipelines.utils.media_io import encode_video
from ltx_pipelines.utils.constants import (
    AUDIO_SAMPLE_RATE,
    DEFAULT_NEGATIVE_PROMPT,
    DEFAULT_VIDEO_GUIDER_PARAMS,
    DEFAULT_AUDIO_GUIDER_PARAMS,
    DEFAULT_FRAME_RATE,
    DEFAULT_NUM_INFERENCE_STEPS,
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
CHECKPOINT_PATH        = "./ltx-2-19b-dev.safetensors"
DISTILLED_LORA_PATH    = "./ltx-2-19b-distilled-lora-384.safetensors"
SPATIAL_UPSAMPLER_PATH = "./ltx-2-spatial-upscaler-x2-1.0.safetensors"
GEMMA_ROOT             = "./gemma-3-12b-local"
PROMPTS_FILE           = "./prompts.txt"
OUTPUT_DIR             = "./outputs"

# ── Generation settings ───────────────────────────────────────────────────────
SEED   = 42
HEIGHT = 1088   # nearest multiple of 64 to 1080 (required for two-stage)
WIDTH  = 1920

# Duration variants: label -> num_frames
# Formula: num_frames = (8 × K) + 1
#   5s  @ 24fps: 120 frames → K=15 → 121
#   8s  @ 24fps: 192 frames → K=24 → 193
#   10s @ 24fps: 240 frames → K=30 → 241
DURATION_VARIANTS = {
    "5s":  121,
    "8s":  193,
    "10s": 241,
}


def load_prompts(path: str) -> list[dict]:
    """Read prompts.txt — one JSON object per line."""
    prompts = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                prompts.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping line {line_no} — invalid JSON: {e}")
    logger.info(f"Loaded {len(prompts)} prompts from {path}")
    return prompts


def output_path_for(source_file: str, suffix: str, output_dir: str) -> str:
    """Convert e.g. '000431.json' + '5s' -> './outputs/000431_5s.mp4'"""
    stem = Path(source_file).stem
    return str(Path(output_dir) / f"{stem}_{suffix}.mp4")


@torch.inference_mode()
def main() -> None:
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    prompts = load_prompts(PROMPTS_FILE)
    if not prompts:
        logger.error("No prompts found — exiting.")
        return

    logger.info("Loading pipeline...")
    pipeline = TI2VidTwoStagesPipeline(
        checkpoint_path=CHECKPOINT_PATH,
        distilled_lora=[
            LoraPathStrengthAndSDOps(
                DISTILLED_LORA_PATH, 1.0, LTXV_LORA_COMFY_RENAMING_MAP
            )
        ],
        spatial_upsampler_path=SPATIAL_UPSAMPLER_PATH,
        gemma_root=GEMMA_ROOT,
        loras=[],
    )
    logger.info("Pipeline loaded.")

    tiling_config = TilingConfig.default()
    total  = len(prompts)
    failed = []  # list of (source_file, suffix) tuples

    for idx, entry in enumerate(prompts, 1):
        source_file = entry.get("source_file", f"unknown_{idx}.json")
        upsampled   = entry.get("upsampled_prompt", "").strip()
        original    = entry.get("prompt", "")

        logger.info(f"[{idx}/{total}] {source_file}")
        logger.info(f"  Prompt: {upsampled[:120]}...")

        if not upsampled:
            logger.warning("  No upsampled_prompt, falling back to prompt.")
            upsampled = original

        if not upsampled:
            logger.error(f"  No prompt available for {source_file}, skipping all durations.")
            for suffix in DURATION_VARIANTS:
                failed.append((source_file, suffix))
            continue

        # Generate each duration variant for this prompt
        for suffix, num_frames in DURATION_VARIANTS.items():
            out_path = output_path_for(source_file, suffix, OUTPUT_DIR)

            logger.info(f"  [{suffix}] {num_frames} frames → {out_path}")

            if Path(out_path).exists():
                logger.info(f"  [{suffix}] Already exists, skipping.")
                continue

            try:
                video, audio = pipeline(
                    prompt=upsampled,
                    negative_prompt=DEFAULT_NEGATIVE_PROMPT,
                    seed=SEED,          # same seed across all durations
                    height=HEIGHT,
                    width=WIDTH,
                    num_frames=num_frames,
                    frame_rate=DEFAULT_FRAME_RATE,
                    num_inference_steps=DEFAULT_NUM_INFERENCE_STEPS,
                    video_guider_params=DEFAULT_VIDEO_GUIDER_PARAMS,
                    audio_guider_params=DEFAULT_AUDIO_GUIDER_PARAMS,
                    images=[],
                    tiling_config=tiling_config,
                )

                encode_video(
                    video=video,
                    fps=DEFAULT_FRAME_RATE,
                    audio=audio,
                    audio_sample_rate=AUDIO_SAMPLE_RATE,
                    output_path=out_path,
                    video_chunks_number=get_video_chunks_number(num_frames, tiling_config),
                )
                logger.info(f"  [{suffix}] Saved: {out_path}")

            except Exception as e:
                logger.error(f"  [{suffix}] FAILED {source_file}: {e}", exc_info=True)
                failed.append((source_file, suffix))

    # Summary
    logger.info("=" * 60)
    total_variants = total * len(DURATION_VARIANTS)
    logger.info(
        f"Done. {total_variants - len(failed)}/{total_variants} videos generated successfully."
    )
    if failed:
        logger.warning(f"Failed ({len(failed)}):")
        for source_file, suffix in failed:
            logger.warning(f"  {source_file} [{suffix}]")


if __name__ == "__main__":
    main()

