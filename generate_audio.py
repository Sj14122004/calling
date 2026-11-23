
from pathlib import Path
import sys
import torch

# Project paths
ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models" / "mithi-v2-flash"
OUTPUT_DIR = ROOT / "public" / "audio"

# Validate model files
if not (MODEL_DIR / "run.py").exists():
    raise FileNotFoundError(
        f"Mithi model not found: {MODEL_DIR}"
    )

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Use CPU with 4 threads
torch.set_num_threads(4)

# Load Mithi once
sys.path.insert(0, str(MODEL_DIR))
from run import Mithi

print("Loading Mithi v2 Flash...")
tts = Mithi(
    model_dir=str(MODEL_DIR),
    device="cpu"
)

# All six calling scripts
PROMPTS = {
    "q1.wav": (
        "नमस्ते, मैं शिवम की असिस्टेंट हूँ। "
        "मेरे बॉस एक सॉफ्टवेयर डेवलपर हैं। "
        "क्या आप अपनी सर्विस के लिए वेबसाइट डिज़ाइन करवाना चाहते हैं, "
        "ताकि आपकी सर्विस की ग्रोथ हो? "
        "हाँ के लिए 1 दबाइए, और ना के लिए 2।"
    ),

    "q2.wav": (
        "क्या आपके पास पहले से वेबसाइट है? "
        "हाँ के लिए 1 दबाइए, और ना के लिए 2।"
    ),

    "q3.wav": (
        "क्या आप अपनी वेबसाइट में सुधार करवाना चाहते हैं? "
        "हाँ के लिए 1 दबाइए, और ना के लिए 2।"
    ),

    "lead.wav": (
        "ठीक है, मैं अपने बॉस को बता दूँगी, "
        "वे आपको कॉल कर लेंगे। धन्यवाद।"
    ),

    "removed.wav": (
        "ठीक है, आपका समय देने के लिए धन्यवाद। "
        "आपका दिन शुभ हो।"
    ),

    "no-response.wav": (
        "कोई जवाब नहीं मिला। धन्यवाद, नमस्ते।"
    ),
}

# Generate all recordings
for filename, text in PROMPTS.items():
    output_path = OUTPUT_DIR / filename

    print(f"\nGenerating: {filename}")

    audio = tts.synthesize(
        text,
        lang="hi",
        voice="ahana"
    )

    tts.save(audio, str(output_path))

    duration = len(audio) / 24000
    size_kb = output_path.stat().st_size / 1024

    print(
        f"Completed: {filename} | "
        f"{duration:.2f} seconds | "
        f"{size_kb:.2f} KB"
    )

print("\nAll six Hindi recordings generated successfully!")
