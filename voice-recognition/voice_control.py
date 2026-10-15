#!/usr/bin/env python3
"""
Voice control for the LeRobot SO101.

Flow: press Enter -> speak -> Whisper transcribes the speech ->
a keyword selects the matching policy -> lerobot-rollout starts.

By default everything runs in test mode: the script only prints which
command it would run. The robot only moves with --live.

Examples:
    python voice_control.py --text-only    # no microphone or Whisper, type commands
    python voice_control.py                # with microphone, test mode
    python voice_control.py --live         # with microphone, robot moves
    python voice_control.py --detection    # check with the camera before each task
    python voice_control.py --replay       # replay a recorded episode instead of a policy

With --detection the robot checks whether the object is actually there
before starting, and answers the question "Was siehst du?".

With --replay the arm replays a recorded episode via lerobot-replay. This works
before any training, but the arm moves blindly: the object must be at exactly
the same spot as in the recorded episode.

Emergency stop during a rollout: Ctrl+C
"""

import argparse
import difflib
import re
import shlex
import subprocess
import sys
from pathlib import Path

# The last camera frame used for detection is saved here with boxes drawn in,
# so you can check what the robot actually saw.
DETECTION_IMAGE = Path(__file__).resolve().parent / "last_detection.jpg"

# object_detection.py lives in the sibling folder object-recognition
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "object-recognition"))

# ---------------------------------------------------------------------------
# CONFIGURATION: adjust here
# ---------------------------------------------------------------------------

HF_USER = "ayah055"
ROBOT_PORT = "COM6"
ROBOT_ID = "follower"
# Must match the camera configuration used during recording exactly!
CAMERAS = "{front: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}}"
ROLLOUT_DURATION = 30  # seconds per run

# Keywords are German and English so both languages work (use --language auto).
# Replies stay German.
# "object" is the COCO class the detector looks for (see GERMAN_NAMES in
# object_detection.py). Set it to None to skip the check for a task.
COMMANDS = {
    "cup": {
        "keywords": ["becher", "tasse", "trinken", "durst", "cup", "drink"],
        "object": "cup",
        "policy": f"{HF_USER}/cup_policy_v2",
        "dataset": f"{HF_USER}/cup_task_v2_20261015_072914",  # used with --replay
        "episode": 3,                                       # used with --replay
        "task": "Bring the cup to the person",
        "reply": "Okay, ich bringe dir den Becher.",
        "missing": "Ich sehe leider keinen Becher.",
    },
    "phone": {
        "keywords": ["handy", "telefon", "smartphone", "phone", "cellphone", "mobile"],
        "object": "cell phone",
        "policy": f"{HF_USER}/phone_policy",
        "dataset": f"{HF_USER}/phone_test",
        "episode": 0,
        "task": "Pick up the phone and put it on the table",
        "reply": "Okay, ich hebe dein Handy auf.",
        "missing": "Ich sehe leider kein Handy.",
    },
}

STOP_WORDS = ["beenden", "ende", "tschüss", "aufhören"]
SEE_PREFIXES = ("sieh", "erkenn")  # "Was siehst du?", "Was erkennst du?"

SAMPLE_RATE = 16000  # Whisper expects 16 kHz

# ---------------------------------------------------------------------------
# Keyword matching
# ---------------------------------------------------------------------------


def normalize(text):
    """Lowercase, strip punctuation, split into words."""
    text = re.sub(r"[^\w\s]", " ", text.lower())
    return text.split()


def word_matches(word, keyword, threshold=0.8):
    """Allows inflected forms (bechers) and small transcription errors (bechr)."""
    if word.startswith(keyword):
        return True
    return difflib.SequenceMatcher(None, word, keyword).ratio() >= threshold


def count_matches(words, keywords):
    return sum(any(word_matches(w, kw) for w in words) for kw in keywords)


def find_command(text):
    """Return (name, status) where status is 'ok', 'none' or 'ambiguous'."""
    words = normalize(text)
    scores = {name: count_matches(words, c["keywords"]) for name, c in COMMANDS.items()}
    scores = {n: s for n, s in scores.items() if s > 0}
    if not scores:
        return None, "none"
    best_score = max(scores.values())
    best = [n for n, s in scores.items() if s == best_score]
    if len(best) > 1:
        return None, "ambiguous"
    return best[0], "ok"


def is_stop(text):
    return count_matches(normalize(text), STOP_WORDS) > 0


def is_see_question(text):
    # Prefix match only, fuzzy matching would confuse "sie" with "sieh"
    return any(w.startswith(p) for w in normalize(text) for p in SEE_PREFIXES)


# ---------------------------------------------------------------------------
# Audio and Whisper (only imported when needed)
# ---------------------------------------------------------------------------


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
    return Whisper(model)


def record(seconds, mic=None):
    import sounddevice as sd

    print(f"Speak now ({seconds} s) ...")
    audio = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                   channels=1, dtype="float32", device=mic)
    sd.wait()
    return audio.flatten()


def loudness(audio):
    import numpy as np

    return float(np.sqrt(np.mean(audio ** 2)))


def transcribe(asr, audio, language="german"):
    return asr.transcribe(audio, language)


# ---------------------------------------------------------------------------
# Camera and object detection
# ---------------------------------------------------------------------------


def camera_objects(camera, confidence):
    """Capture a frame and run detection. Returns None if the camera fails."""
    from object_detection import capture_frame, detect

    try:
        objects = detect(capture_frame(camera), confidence, save_path=DETECTION_IMAGE)
    except RuntimeError as e:
        print(f"Camera error: {e}")
        return None
    print(f"  (camera frame saved to {DETECTION_IMAGE})")
    return objects


def what_do_you_see(camera, confidence):
    from object_detection import describe

    objects = camera_objects(camera, confidence)
    if objects is not None:
        print(f"> {describe(objects)}")


def object_present(command, camera, confidence):
    """True if the task's object is in view (or the task has no object to check)."""
    if not command.get("object"):
        return True
    from object_detection import find

    objects = camera_objects(camera, confidence)
    if objects is None:
        return False
    matches = find(objects, command["object"])
    if not matches:
        print(f"> {command['missing']}")
        return False
    print(f"  Detected: {matches[0]['name']} (confidence {matches[0]['confidence']:.2f})")
    return True


# ---------------------------------------------------------------------------
# Robot
# ---------------------------------------------------------------------------


def rollout_command(command):
    return [
        "lerobot-rollout",
        "--strategy.type=base",
        f"--policy.path={command['policy']}",
        "--robot.type=so101_follower",
        f"--robot.port={ROBOT_PORT}",
        f"--robot.id={ROBOT_ID}",
        f"--robot.cameras={CAMERAS}",
        f"--task={command['task']}",
        f"--duration={ROLLOUT_DURATION}",
    ]


def replay_command(command):
    return [
        "lerobot-replay",
        "--robot.type=so101_follower",
        f"--robot.port={ROBOT_PORT}",
        f"--robot.id={ROBOT_ID}",
        f"--dataset.repo_id={command['dataset']}",
        f"--dataset.episode={command['episode']}",
    ]


def execute(name, live, detection=None, replay=False):
    """detection is None or a tuple (camera_index, confidence)."""
    command = COMMANDS[name]
    if detection and not object_present(command, *detection):
        return
    print(f"> {command['reply']}")
    cmd = replay_command(command) if replay else rollout_command(command)
    if not live:
        print("[test mode] would run:\n  " + " ".join(shlex.quote(c) for c in cmd))
        return
    try:
        subprocess.run(cmd, check=True)
        print("Done.")
    except KeyboardInterrupt:
        print("\nRollout aborted.")
    except subprocess.CalledProcessError as e:
        print(f"Rollout failed (exit code {e.returncode}).")
    except FileNotFoundError:
        print("lerobot-rollout not found. Is LeRobot installed and the environment active?")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Voice control for LeRobot")
    parser.add_argument("--live", action="store_true",
                        help="Actually run rollouts (otherwise test mode)")
    parser.add_argument("--text-only", action="store_true",
                        help="No microphone or Whisper, typed commands only")
    parser.add_argument("--model", default="openai/whisper-small",
                        help="Whisper model from the Hugging Face Hub")
    parser.add_argument("--mic", type=int, default=None,
                        help="Microphone number (see speech_to_text.py --list-mics)")
    parser.add_argument("--language", default="german",
                        help='Spoken language, or "auto" for mixed German/English commands')
    parser.add_argument("--seconds", type=float, default=4.0,
                        help="Recording length per command")
    parser.add_argument("--silence", type=float, default=0.01,
                        help="Loudness threshold below which a recording counts as empty")
    parser.add_argument("--detection", action="store_true",
                        help="Check with the camera before each task whether the object is there")
    parser.add_argument("--camera", type=int, default=2,
                        help="Camera index for detection (same camera as in CAMERAS)")
    parser.add_argument("--confidence", type=float, default=0.4,
                        help="Minimum confidence for object detection (0 to 1)")
    parser.add_argument("--replay", action="store_true",
                        help="Replay a recorded episode instead of running a trained policy")
    args = parser.parse_args()
    detection = (args.camera, args.confidence) if args.detection else None

    asr = None if args.text_only else load_whisper(args.model)
    if args.detection:
        from object_detection import load_model

        print("Loading object detection ...")
        load_model()

    print("\nAvailable tasks:")
    for name, c in COMMANDS.items():
        print(f"  {name}: keywords {', '.join(c['keywords'])}")
    print("Mode:", "LIVE (robot will move!)" if args.live else "test",
          "| replaying recorded episodes" if args.replay else "| trained policies")
    if args.detection:
        print('Object detection enabled. You can also ask: "Was siehst du?"')

    while True:
        try:
            user_input = input("\n[Enter] to speak, type text, or 'q' to quit: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user_input.lower() == "q":
            break

        if user_input:
            text = user_input
        elif args.text_only:
            continue
        else:
            audio = record(args.seconds, args.mic)
            level = loudness(audio)
            if level < args.silence:
                print(f"Heard nothing (level {level:.4f}). Speak louder or lower --silence.")
                continue
            text = transcribe(asr, audio, args.language)

        print(f'Heard: "{text}"')

        if is_stop(text):
            break
        name, status = find_command(text)
        if status != "ok" and is_see_question(text):
            if detection:
                what_do_you_see(*detection)
            else:
                print("Object detection is off. Start with --detection.")
            continue
        if status == "none":
            print("> Das kann ich leider nicht.")
            print("  Known tasks: " + ", ".join(COMMANDS))
        elif status == "ambiguous":
            print("> Das war nicht eindeutig. Bitte nur eine Aufgabe nennen.")
        else:
            execute(name, args.live, detection, args.replay)

    print("Bye!")


if __name__ == "__main__":
    main()