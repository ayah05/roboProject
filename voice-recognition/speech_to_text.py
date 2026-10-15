#!/usr/bin/env python3
"""
Speech to text with Whisper, for testing and tuning the voice input on its own.

Press Enter, speak, and the script prints what Whisper understood, how loud the
recording was and how long the transcription took. No robot or camera needed.

Examples:
    python speech_to_text.py --list-mics            # show available microphones
    python speech_to_text.py                        # default mic, whisper-small
    python speech_to_text.py --mic 1                # use microphone number 1
    python speech_to_text.py --model openai/whisper-base   # faster, less accurate
    python speech_to_text.py --save                 # also save each recording as WAV
    python speech_to_text.py --language auto        # detect German or English automatically

Installation:
    pip install transformers sounddevice numpy
"""

import argparse
import time
import wave
from pathlib import Path

SAMPLE_RATE = 16000  # Whisper expects 16 kHz mono audio


def list_microphones():
    import sounddevice as sd

    default_input = sd.default.device[0]
    print("Available microphones:")
    for i, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0:
            marker = "  (default)" if i == default_input else ""
            print(f"  {i}: {dev['name']}{marker}")


class Whisper:
    """Whisper without the transformers pipeline. The pipeline tries to import
    torchcodec, which fails on Windows without FFmpeg DLLs. We pass raw audio
    from the microphone directly, so no audio decoding library is needed."""

    def __init__(self, model):
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        print(f"Loading {model} on {self.device} (downloaded on first run) ...")
        self.processor = WhisperProcessor.from_pretrained(model)
        self.model = WhisperForConditionalGeneration.from_pretrained(model).to(self.device)
        self.model.eval()

    def transcribe(self, audio, language="german"):
        """language="auto" lets Whisper detect the language itself, which works
        better when you mix German and English commands."""
        import torch

        inputs = self.processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        features = inputs.input_features.to(self.device)
        options = {"task": "transcribe"}
        if language != "auto":
            options["language"] = language
        with torch.no_grad():
            ids = self.model.generate(features, **options)
        return self.processor.batch_decode(ids, skip_special_tokens=True)[0].strip()


def load_whisper(model):
    start = time.time()
    asr = Whisper(model)
    print(f"Model ready after {time.time() - start:.1f} s")
    return asr


def record(seconds, mic=None):
    import sounddevice as sd

    print(f"Speak now ({seconds:.0f} s) ...")
    audio = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                   channels=1, dtype="float32", device=mic)
    sd.wait()
    return audio.flatten()


def loudness(audio):
    import numpy as np

    return float(np.sqrt(np.mean(audio ** 2)))


def save_wav(audio, path):
    """Save float32 audio as a 16-bit WAV file (no extra dependencies)."""
    import numpy as np

    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SAMPLE_RATE)
        f.writeframes(pcm.tobytes())


def transcribe(asr, audio, language):
    return asr.transcribe(audio, language)


def main():
    parser = argparse.ArgumentParser(description="Whisper speech-to-text test")
    parser.add_argument("--list-mics", action="store_true",
                        help="List available microphones and exit")
    parser.add_argument("--mic", type=int, default=None,
                        help="Microphone number from --list-mics (default: system default)")
    parser.add_argument("--model", default="openai/whisper-small",
                        help="Whisper model from the Hugging Face Hub")
    parser.add_argument("--language", default="german",
                        help='Spoken language, or "auto" to detect it (for mixed German/English)')
    parser.add_argument("--seconds", type=float, default=4.0,
                        help="Recording length per command")
    parser.add_argument("--silence", type=float, default=0.01,
                        help="Loudness threshold below which a recording counts as empty")
    parser.add_argument("--save", action="store_true",
                        help="Save each recording as recording_N.wav next to this script")
    args = parser.parse_args()

    if args.list_mics:
        list_microphones()
        return

    asr = load_whisper(args.model)
    out_dir = Path(__file__).resolve().parent
    count = 0

    while True:
        try:
            cmd = input("\n[Enter] to record, 'q' to quit: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break
        if cmd == "q":
            break

        audio = record(args.seconds, args.mic)
        level = loudness(audio)
        count += 1

        if args.save:
            path = out_dir / f"recording_{count}.wav"
            save_wav(audio, path)
            print(f"  saved to {path}")

        if level < args.silence:
            print(f"  Heard nothing (level {level:.4f}, threshold {args.silence}).")
            print("  Speak louder, pick another --mic or lower --silence.")
            continue

        start = time.time()
        text = transcribe(asr, audio, args.language)
        print(f'  Text:  "{text}"')
        print(f"  Level: {level:.4f}   Transcription time: {time.time() - start:.1f} s")

    print("Bye!")


if __name__ == "__main__":
    main()