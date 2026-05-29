#!/usr/bin/env python3
"""
v17 Coder Swarm Member #1 — A/B PoC Harness + Tensor Diff Validation

Self-contained validation driver for the geometry input ablation added to
scripts/inference/infer.py (--geometry-mode {A,B} with divergent builders).

PRIMARY GOAL: On identical prompt + image, produce the two geometry_encoder_inputs
tensors (A vs B), compute+print rich diff statistics, optionally save them + a
report. Secondary: optionally shell out to the real infer.py for A/B text outputs
so you get measurable, diffable (input tensors + generated responses) artifacts
for quick PoC validation before heavier lmms_eval runs.

FAST PATH (recommended for quick validation, minimal VRAM/RAM, works on any
torch+transformers box including stock ICRN or MSI 5080):
  python scripts/inference/run_v17_geometry_ablation.py \
    --image assets/office_chairs.jpg \
    --prompt "You are a navigation agent. Analyze the 3D scene layout and output 3-5 safe waypoints as (x,y,z) tuples to reach the target." \
    --output-dir runlogs/v17_ablation_$(date +%H%M%S)

  # No full model weights loaded. Only the processor (~few hundred MB download
  # first time) + pure tensor construction for A/B builders. Finishes in <30s.

FULL PATH (end-to-end, uses the actual infer.py codepath + model weights):
  add --run-inference
  (Sequential A then B generate; ~model_size VRAM; responses captured in report.)

Outputs in <output-dir>/ :
  - v17_geometry_ablation_report.json   (all stats, prompt, shapes, responses if any)
  - geometry_A.pt / geometry_B.pt       (if --save-tensors; the exact stacked tensors
                                          that would be passed as geometry_encoder_inputs)
  - infer_response_A.json / _B.json     (if --run-inference; the real infer outputs)

The A/B builders (from current infer.py):
  A = bicubic resize + pad=1.0 (white) + /255.0
  B = nearest resize + pad=0.0 (black) + (/255.0 * 0.9)
These are deliberately divergent for the Stage 1 PoC to surface whether the
geometry encoder is sensitive to input formulation on ag scenes.

Runnable with minimal deps (tensor-only path):
  torch, torchvision (optional), transformers>=5.3, Pillow, numpy
  (qwen_vl_utils + decord + SpatialStack only needed for --run-inference full path
   or if you want dynamic import of the exact builder funcs.)

H200 / MSI 5080 ready. 0% util + 17GiB headroom = perfect for this.

Part of Kulbir CoRL war room v17 geometry ablation swarm (parallel with #2 lmms).
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor


# --- Fallback builders (exact match to v17 edits in infer.py + lmms qwen3_5.py) ---
# These are pure and always available even if dynamic import of infer.py fails
# (e.g. missing qwen_vl_utils in a throwaway torch env). Keeps harness self-contained.

def build_qwen3_5_geometry_inputs_A(images, image_grid_thw, patch_size: int = 14):
    """Variant A (baseline-style): bicubic resize, pad=1.0 (white), /255 norm."""
    geometry_tensors = []
    max_height = 0
    max_width = 0

    for image, grid in zip(images, image_grid_thw):
        _, grid_h, grid_w = [int(v) for v in grid.tolist()]
        target_height = grid_h * patch_size
        target_width = grid_w * patch_size
        resized = image.resize((target_width, target_height), Image.Resampling.BICUBIC)
        tensor = torch.from_numpy(np.array(resized, copy=True)).permute(2, 0, 1).float() / 255.0
        geometry_tensors.append(tensor)
        max_height = max(max_height, target_height)
        max_width = max(max_width, target_width)

    padded_tensors = []
    for tensor in geometry_tensors:
        h_padding = max_height - tensor.shape[1]
        w_padding = max_width - tensor.shape[2]
        if h_padding > 0 or w_padding > 0:
            pad_top = h_padding // 2
            pad_bottom = h_padding - pad_top
            pad_left = w_padding // 2
            pad_right = w_padding - pad_left
            tensor = torch.nn.functional.pad(
                tensor, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
            )
        padded_tensors.append(tensor)

    return padded_tensors


def build_qwen3_5_geometry_inputs_B(images, image_grid_thw, patch_size: int = 14):
    """Variant B (diff test for PoC): nearest resize, pad=0.0 (black), /255 * 0.9."""
    geometry_tensors = []
    max_height = 0
    max_width = 0

    for image, grid in zip(images, image_grid_thw):
        _, grid_h, grid_w = [int(v) for v in grid.tolist()]
        target_height = grid_h * patch_size
        target_width = grid_w * patch_size
        resized = image.resize((target_width, target_height), Image.Resampling.NEAREST)
        tensor = torch.from_numpy(np.array(resized, copy=True)).permute(2, 0, 1).float() / 255.0
        tensor = tensor * 0.9
        geometry_tensors.append(tensor)
        max_height = max(max_height, target_height)
        max_width = max(max_width, target_width)

    padded_tensors = []
    for tensor in geometry_tensors:
        h_padding = max_height - tensor.shape[1]
        w_padding = max_width - tensor.shape[2]
        if h_padding > 0 or w_padding > 0:
            pad_top = h_padding // 2
            pad_bottom = h_padding - pad_top
            pad_left = w_padding // 2
            pad_right = w_padding - pad_left
            tensor = torch.nn.functional.pad(
                tensor, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0.0
            )
        padded_tensors.append(tensor)

    return padded_tensors


# --- Lightweight dynamic import of real builders (when env has full deps) ---
def try_load_builders_from_infer():
    """Attempt to pull the exact live versions from the edited infer.py.
    Returns (builder_A, builder_B) or (None, None) on any failure.
    """
    candidates = [
        Path(__file__).resolve().parent / "infer.py",
        Path("/home/ksa5/corl/SpatialStack/scripts/inference/infer.py"),
        Path("/home/ksa5/SpatialStack/scripts/inference/infer.py"),
    ]
    for infer_path in candidates:
        if infer_path.exists():
            try:
                import importlib.util
                spec = importlib.util.spec_from_file_location("_v17_infer", str(infer_path))
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                a = getattr(mod, "build_qwen3_5_geometry_inputs_A", None)
                b = getattr(mod, "build_qwen3_5_geometry_inputs_B", None)
                if a and b:
                    print(f"[info] Using live builders from {infer_path}")
                    return a, b
            except Exception as e:
                print(f"[warn] Dynamic import of builders from {infer_path} failed: {e}")
    return None, None


def load_pil_images(image_path: str):
    p = Path(image_path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if p.is_file():
        return [Image.open(p).convert("RGB")]
    # Future-proof: could extend to dir/video like load_visuals, but PoC keeps single-image
    raise ValueError(f"Harness PoC supports single image file only (got dir?): {image_path}")


def get_image_grid_thw(processor, pil_images, max_pixels: int, min_pixels: int):
    """Replicate the exact processor call used in infer.py qwen3_5 geometry path
    (text + images=raw_pils) to obtain the authoritative image_grid_thw.
    """
    # Build a minimal chat-style text that the processor will accept.
    # The grid_thw depends on the actual tokenization + image processor sizing.
    img_tokens = "<|vision_start|><|image_pad|><|vision_end|>" * len(pil_images)
    text = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{img_tokens}{ ' dummy for grid' }<|im_end|>\n<|im_start|>assistant\n"

    proc_inputs = processor(
        text=[text],
        images=pil_images,
        videos=None,
        padding=True,
        return_tensors="pt",
    )
    if "image_grid_thw" not in proc_inputs:
        raise RuntimeError("Processor did not return image_grid_thw. Is this a Qwen VL processor?")
    return proc_inputs["image_grid_thw"]


def compute_tensor_stats(t: torch.Tensor, label: str):
    t = t.detach().float().cpu()
    return {
        "label": label,
        "shape": tuple(t.shape),
        "mean": float(t.mean()),
        "std": float(t.std(unbiased=False)),
        "min": float(t.min()),
        "max": float(t.max()),
        "l2_norm": float(torch.linalg.norm(t)),
        "numel": int(t.numel()),
    }


def main():
    parser = argparse.ArgumentParser(
        description="v17 Coder Swarm #1: A/B geometry input tensor diff + optional infer runs"
    )
    parser.add_argument(
        "--model-path",
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        help="HF repo for processor (must support image_grid_thw; any Qwen2/2.5/3.5 VL works for input tensor PoC)",
    )
    parser.add_argument("--image", required=True, help="Single image path (assets/*.jpg recommended for ag-like scenes)")
    parser.add_argument(
        "--prompt",
        default="You are navigating a 3D indoor scene as a mobile agent. Output a short list of 3D waypoints (x y z) to safely reach the main target object while avoiding obstacles.",
        help="Text prompt (used in report + passed to --run-inference calls)",
    )
    parser.add_argument("--max-pixels", type=int, default=1605632)
    parser.add_argument("--min-pixels", type=int, default=256 * 28 * 28)
    parser.add_argument(
        "--output-dir",
        default="",
        help="Directory for report + artifacts (default: /home/ksa5/runlogs/v17_geo_ablation_YYYYMMDD_HHMMSS)",
    )
    parser.add_argument("--save-tensors", action="store_true", help="Write geometry_A.pt and geometry_B.pt (stacked C,H,W tensors)")
    parser.add_argument(
        "--run-inference",
        action="store_true",
        help="Shell out to infer.py --geometry-mode A and B (real model forward passes + generated text). Requires full SpatialStack env + deps.",
    )
    parser.add_argument("--infer-script", default="", help="Explicit path to infer.py (auto-detected otherwise)")
    parser.add_argument("--device", default="cuda:0", help="Only used for --run-inference subprocesses")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = parser.parse_args()

    print("=== v17 Coder Swarm Member #1: A/B Geometry Input Tensor Diff Validation Harness ===")
    print(f"Time: {datetime.now().isoformat()}")
    print(f"Processor source: {args.model_path}")
    print(f"Test image: {args.image}")
    print(f"Prompt (truncated): {args.prompt[:100]}...")

    # Output layout
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("/home/ksa5/runlogs") / f"v17_geo_ablation_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Artifacts -> {out_dir}")

    # 1. Load visuals (pure PIL)
    pil_images = load_pil_images(args.image)
    print(f"Loaded {len(pil_images)} PIL image(s) for A/B construction")

    # 2. Processor (light) -> authoritative grid_thw for this model + resolution config
    print("Loading AutoProcessor (first run may fetch ~100-300 MB processor files)...")
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        max_pixels=args.max_pixels,
        min_pixels=args.min_pixels,
        padding_side="left",
    )
    image_grid_thw = get_image_grid_thw(processor, pil_images, args.max_pixels, args.min_pixels)
    print(f"image_grid_thw (from processor): {image_grid_thw.tolist()}")

    # 3. Obtain builders (prefer live from infer.py, fall back to embedded copies)
    builder_A, builder_B = try_load_builders_from_infer()
    if builder_A is None or builder_B is None:
        print("[info] Using embedded fallback builders (identical to current infer.py v17 code)")
        builder_A = build_qwen3_5_geometry_inputs_A
        builder_B = build_qwen3_5_geometry_inputs_B

    # 4. Build the two geometry input tensors exactly as infer.py does for --geometry-mode A/B
    geo_list_A = builder_A(pil_images, image_grid_thw)
    geo_list_B = builder_B(pil_images, image_grid_thw)
    geo_A = torch.stack(geo_list_A) if geo_list_A else torch.empty(0)
    geo_B = torch.stack(geo_list_B) if geo_list_B else torch.empty(0)

    print(f"geometry_A stacked: {tuple(geo_A.shape)}")
    print(f"geometry_B stacked: {tuple(geo_B.shape)}")

    # 5. Rich statistics + diff
    stats_A = compute_tensor_stats(geo_A, "A")
    stats_B = compute_tensor_stats(geo_B, "B")

    delta = (geo_A - geo_B).abs()
    diff_stats = {
        "max_abs_diff": float(delta.max()),
        "mean_abs_diff": float(delta.mean()),
        "std_abs_diff": float(delta.std(unbiased=False)),
        "l2_norm_of_diff": float(torch.linalg.norm(geo_A - geo_B)),
        "relative_l2_diff": float(torch.linalg.norm(geo_A - geo_B) / (torch.linalg.norm(geo_A) + 1e-8)),
        "num_elements": int(delta.numel()),
        "elements_gt_1e-5": int((delta > 1e-5).sum()),
        "elements_gt_0_01": int((delta > 0.01).sum()),
        "elements_gt_0_1": int((delta > 0.1).sum()),
    }

    print("\n================= GEOMETRY INPUT TENSOR DIFF REPORT =================")
    print("A (bicubic + pad=1.0 + /255):")
    print(json.dumps(stats_A, indent=2))
    print("\nB (nearest + pad=0.0 + /255*0.9):")
    print(json.dumps(stats_B, indent=2))
    print("\nA vs B absolute differences:")
    print(json.dumps(diff_stats, indent=2))
    print("==================================================================\n")

    # 6. Assemble machine-readable report
    report = {
        "harness_role": "v17 Coder Swarm Member #1 (A/B PoC harness + tensor diff validation)",
        "timestamp": datetime.now().isoformat(),
        "model_path_for_processor": args.model_path,
        "image": str(Path(args.image).resolve()),
        "prompt": args.prompt,
        "geometry_input_shapes": [list(geo_A.shape), list(geo_B.shape)],
        "stats_A": stats_A,
        "stats_B": stats_B,
        "diff_stats": diff_stats,
        "builder_description": {
            "A": "bicubic resize, constant pad value=1.0, values / 255.0",
            "B": "nearest resize, constant pad value=0.0, values / 255.0 * 0.9 (deliberate contrast+interp shift)",
        },
        "note": "These geometry tensors are what get passed to the geometry encoder (VGGT/PI3 etc.) under --geometry-mode A vs B. Large diff here validates the ablation harness is live.",
    }

    if args.save_tensors:
        torch.save(geo_A, out_dir / "geometry_A.pt")
        torch.save(geo_B, out_dir / "geometry_B.pt")
        report["saved_tensor_files"] = ["geometry_A.pt", "geometry_B.pt"]
        print(f"[saved] {out_dir / 'geometry_A.pt'} and geometry_B.pt")

    # 7. Optional: real end-to-end A/B runs via the canonical infer.py (for response diffs)
    responses = {}
    if args.run_inference:
        print("=== --run-inference: launching real infer.py for modes A and B ===")
        infer_script = args.infer_script or str((Path(__file__).parent / "infer.py").resolve())
        if not Path(infer_script).exists():
            print(f"[error] Could not find infer.py at {infer_script} — skipping full inference runs.")
        else:
            for mode in ("A", "B"):
                resp_json = out_dir / f"infer_response_mode{mode}.json"
                cmd = [
                    sys.executable,
                    infer_script,
                    "--model-path",
                    args.model_path,
                    "--image",
                    args.image,
                    "--prompt",
                    args.prompt,
                    "--geometry-mode",
                    mode,
                    "--output-json",
                    str(resp_json),
                    "--max-new-tokens",
                    "256",
                    "--temperature",
                    "0.0",
                    "--device",
                    args.device,
                    "--dtype",
                    args.dtype,
                ]
                print(f"  [mode {mode}] {' '.join(map(str, cmd))}")
                try:
                    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
                    stdout_tail = proc.stdout.strip()[-400:] if proc.stdout else ""
                    if proc.returncode != 0:
                        print(f"    WARNING: exit={proc.returncode} stderr_tail={proc.stderr[-300:]}")
                    if resp_json.exists():
                        with open(resp_json) as f:
                            responses[mode] = json.load(f)
                    else:
                        responses[mode] = {
                            "response": stdout_tail or "(no stdout)",
                            "stderr_tail": proc.stderr[-400:] if proc.stderr else "",
                            "returncode": proc.returncode,
                        }
                except subprocess.TimeoutExpired:
                    responses[mode] = {"error": "timeout after 600s"}
                except Exception as ex:
                    responses[mode] = {"error": str(ex)}
            report["full_inference_responses"] = responses
            print("Full A/B inference responses captured.")

    # Finalize
    report_path = out_dir / "v17_geometry_ablation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n[done] Report written: {report_path}")
    if responses:
        print("Responses (A vs B) side-by-side preview:")
        for m in ("A", "B"):
            r = responses.get(m, {})
            txt = (r.get("response") or str(r))[:220].replace("\n", " ")
            print(f"  {m}: {txt}...")

    print("\n=== Swarm #1 harness finished. Feed the report + .pt files into your tensor-diff / figure pipeline. ===")


if __name__ == "__main__":
    main()
