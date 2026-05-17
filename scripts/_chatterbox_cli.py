"""Chatterbox CLI — runs Resemble AI Chatterbox TTS. Invoked via scripts/chatterbox-cli.sh.

Default model is Chatterbox-Turbo (350M, distilled one-step decoder, ~6x real-time on GPU,
supports paralinguistic tags like [laugh], [cough], [chuckle] inline in text).
Use --model base for the original 500M model (smoother emotion exaggeration).
"""
import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS
from chatterbox.tts_turbo import ChatterboxTurboTTS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = PROJECT_ROOT / "output" / f"out_{datetime.now():%Y%m%d_%H%M%S}.wav"

# Model-specific defaults from each class's generate() signature.
MODEL_DEFAULTS = {
    "turbo": {"exaggeration": 0.0, "cfg_weight": 0.0, "temperature": 0.8},
    "base":  {"exaggeration": 0.5, "cfg_weight": 0.5, "temperature": 0.8},
}

p = argparse.ArgumentParser(
    description="Chatterbox TTS CLI (Resemble AI). Default model: turbo.",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog=(
        "Turbo accepts paralinguistic tags inline, e.g.:\n"
        '  ./chatterbox-cli.sh "That was hilarious [laugh] really."'
    ),
)
p.add_argument("text", nargs="?", help="Text to speak ('-' = read stdin)")
p.add_argument("-o", "--out", default=str(DEFAULT_OUT),
               help="Output WAV path (default: <project>/output/out_YYYYMMDD_HHMMSS.wav)")
p.add_argument("-m", "--model", default="turbo", choices=("turbo", "base"),
               help="turbo (default, faster, supports [laugh]/[cough] tags) or base (original 500M)")
p.add_argument("-r", "--ref", default=None,
               help="Reference WAV for zero-shot voice cloning")
p.add_argument("-e", "--exaggeration", type=float, default=None,
               help="Emotion exaggeration (turbo default 0.0; base default 0.5; higher = more expressive)")
p.add_argument("-c", "--cfg-weight", type=float, default=None,
               help="CFG weight (turbo default 0.0; base default 0.5)")
p.add_argument("-t", "--temperature", type=float, default=None, help="Sampling temperature (default 0.8)")
p.add_argument("--device", default=None, choices=["cpu", "mps", "cuda"],
               help="Override device (auto-detected if omitted)")
args = p.parse_args()

text = args.text
if text in (None, "-"):
    text = sys.stdin.read()
text = (text or "").strip()
if not text:
    sys.exit("error: empty text")

device = args.device or ("mps" if torch.backends.mps.is_available()
                        else "cuda" if torch.cuda.is_available() else "cpu")

# Patch torch.load to default map_location -> device (per upstream example_for_mac.py)
_orig_torch_load = torch.load
def _patched(*a, **kw):
    if "map_location" not in kw:
        kw["map_location"] = torch.device(device)
    return _orig_torch_load(*a, **kw)
torch.load = _patched

ModelCls = ChatterboxTurboTTS if args.model == "turbo" else ChatterboxTTS
print(f"[chatterbox] model={args.model} device={device}", file=sys.stderr)
t0 = time.time()
model = ModelCls.from_pretrained(device=device)
print(f"[chatterbox] loaded in {time.time()-t0:.2f}s (sr={model.sr})", file=sys.stderr)

# Apply model-specific defaults when the user didn't pass a value.
defaults = MODEL_DEFAULTS[args.model]
gen_kwargs = {
    "exaggeration": args.exaggeration if args.exaggeration is not None else defaults["exaggeration"],
    "cfg_weight":   args.cfg_weight   if args.cfg_weight   is not None else defaults["cfg_weight"],
    "temperature":  args.temperature  if args.temperature  is not None else defaults["temperature"],
}
if args.ref:
    gen_kwargs["audio_prompt_path"] = args.ref

t0 = time.time()
wav = model.generate(text, **gen_kwargs)
if wav.dim() == 1:
    wav = wav.unsqueeze(0)
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
ta.save(args.out, wav.detach().cpu(), model.sr)
dur = wav.shape[-1] / model.sr
gen = time.time() - t0
print(f"[chatterbox] {dur:.2f}s audio in {gen:.2f}s (RTF={gen/dur:.3f}) -> {args.out}",
      file=sys.stderr)
