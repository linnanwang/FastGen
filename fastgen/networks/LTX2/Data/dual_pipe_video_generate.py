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
    DEFAULT_NUM_FRAMES,
    DEFAULT_FRAME_RATE,
    DEFAULT_NUM_INFERENCE_STEPS,
)

# ── Paths ────────────────────────────────────────────────────────────────────
CHECKPOINT_PATH        = "./ltx-2-19b-dev.safetensors"
DISTILLED_LORA_PATH    = "./ltx-2-19b-distilled-lora-384.safetensors"
SPATIAL_UPSAMPLER_PATH = "./ltx-2-spatial-upscaler-x2-1.0.safetensors"
GEMMA_ROOT             = "./gemma-3-12b-local"
OUTPUT_PATH            = "./output_1080p.mp4"

# ── Generation settings ──────────────────────────────────────────────────────
PROMPT = "The camera opens in a calm, sunlit frog yoga studio. Warm morning light washes over the wooden floor as incense smoke drifts lazily in the air. The senior frog instructor sits cross-legged at the center, eyes closed, voice deep and calm. “We are one with the pond.” All the frogs answer softly: “Ommm…” “We are one with the mud.” “Ommm…” He smiles faintly. “We are one with the flies.” A pause. The camera pans to the side towards one frog who twitches, eyes darting. Suddenly its tongue snaps out, catching a fly mid-air and pulling it into its mouth. The master exhales slowly, still serene. “But we do not chase the flies…” Beat. “not during class.” The guilty frog lowers its head in shame, folding its hands back into a meditative pose. The other frogs resume their chant: “Ommm…” Camera holds for a moment on the embarrassed frog, eyes closed too tightly, pretending nothing happened."
#"EXT. SMALL TOWN STREET – MORNING – LIVE NEWS BROADCAST The shot opens on a news reporter standing in front of a row of cordoned-off cars, yellow caution tape fluttering behind him. The light is warm, early sun reflecting off the camera lens. The faint hum of chatter and distant drilling fills the air. The reporter, composed but visibly excited, looks directly into the camera, microphone in hand. Reporter (live): “Thank you, Sylvia. And yes — this is a sentence I never thought I’d say on live television — but this morning, here in the quiet town of New Castle, Vermont… black gold has been found!” He gestures slightly toward the field behind him. Reporter (grinning): “If my cameraman can pan over, you’ll see what all the excitement’s about.” The camera pans right, slowly revealing a construction site surrounded by workers in hard hats. A beat of silence — then, with a sudden roar, a geyser of oil erupts from the ground, blasting upward in a violent plume. Workers cheer and scramble, the black stream glistening in the morning light. The camera shakes slightly, trying to stay focused through the chaos. Reporter (off-screen, shouting over the noise): “There it is, folks — the moment New Castle will never forget!” The camera catches the sunlight gleaming off the oil mist before pulling back, revealing the entire scene — the small-town skyline silhouetted against the wild fountain of oil."
SEED   = 20173261
HEIGHT = 1088   # nearest multiple of 64 to 1080 (required for two-stage)
WIDTH  = 1920


@torch.inference_mode()
def main() -> None:
    # Build pipeline
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

    tiling_config = TilingConfig.default()

    # Generate
    video, audio = pipeline(
        prompt=PROMPT,
        negative_prompt=DEFAULT_NEGATIVE_PROMPT,
        seed=SEED,
        height=HEIGHT,
        width=WIDTH,
        num_frames=482, # DEFAULT_NUM_FRAMES,
        frame_rate=DEFAULT_FRAME_RATE,
        num_inference_steps=DEFAULT_NUM_INFERENCE_STEPS,
        video_guider_params=DEFAULT_VIDEO_GUIDER_PARAMS,
        audio_guider_params=DEFAULT_AUDIO_GUIDER_PARAMS,
        images=[],
        tiling_config=tiling_config,
    )

    # Encode and save
    encode_video(
        video=video,
        fps=DEFAULT_FRAME_RATE,
        audio=audio,
        audio_sample_rate=AUDIO_SAMPLE_RATE,
        output_path=OUTPUT_PATH,
        video_chunks_number=get_video_chunks_number(DEFAULT_NUM_FRAMES, tiling_config),
    )
    print(f"Video saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
