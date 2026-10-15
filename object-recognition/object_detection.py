#!/usr/bin/env python3
"""
Object detection for the assistive robot project.

Uses YOLO (pretrained on the COCO dataset with 80 everyday objects such as
cup, bottle, cell phone, remote, book and scissors). No training required,
the model weights are downloaded automatically on first use.

Examples:
    python object_detection.py --live            # show camera feed with boxes, press q to quit
    python object_detection.py                   # take one photo and list detected objects
    python object_detection.py --image photo.jpg # test with a saved image

Installation:
    pip install ultralytics opencv-python
"""

import argparse
import time
from collections import Counter

# COCO class names -> German names for console output and spoken replies
GERMAN_NAMES = {
    "cup": "Becher", "bottle": "Flasche", "wine glass": "Glas", "bowl": "Schüssel",
    "cell phone": "Handy", "remote": "Fernbedienung", "book": "Buch",
    "scissors": "Schere", "spoon": "Löffel", "fork": "Gabel", "knife": "Messer",
    "toothbrush": "Zahnbürste", "mouse": "Maus", "keyboard": "Tastatur",
    "laptop": "Laptop", "clock": "Uhr", "vase": "Vase", "teddy bear": "Teddybär",
    "sports ball": "Ball", "banana": "Banane", "apple": "Apfel", "orange": "Orange",
    "person": "Person",
}

_model = None


def load_model(name="yolo11n.pt"):
    """Load YOLO once. The 'n' variant is the smallest and fastest."""
    global _model
    if _model is None:
        from ultralytics import YOLO

        _model = YOLO(name)
    return _model


def capture_frame(index=0, warmup_s=1.0):
    """Capture a single frame and release the camera right away,
    so lerobot-rollout can use it afterwards."""
    import cv2

    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {index}.")
    try:
        # Webcams need a moment for auto exposure and white balance. The first
        # frames are often dark, so keep reading for warmup_s seconds.
        ok, frame = cap.read()
        end = time.time() + warmup_s
        while time.time() < end:
            ok, frame = cap.read()
    finally:
        cap.release()
    if not ok:
        raise RuntimeError("Could not read a frame from the camera.")
    return frame


def detect(frame, min_confidence=0.4, save_path=None):
    """Return a list of detected objects, each with its COCO name, German name,
    confidence and bounding box [x1, y1, x2, y2] in pixels.
    If save_path is given, the frame is saved there with boxes drawn in."""
    result = load_model()(frame, conf=min_confidence, verbose=False)[0]
    if save_path:
        import cv2

        cv2.imwrite(str(save_path), result.plot())
    objects = []
    for box in result.boxes:
        name = result.names[int(box.cls)]
        objects.append({
            "name": name,
            "german": GERMAN_NAMES.get(name, name),
            "confidence": float(box.conf),
            "box": [round(v) for v in box.xyxy[0].tolist()],
        })
    return objects


def find(objects, name):
    """All detected objects with the given COCO name, best match first."""
    matches = [o for o in objects if o["name"] == name]
    return sorted(matches, key=lambda o: o["confidence"], reverse=True)


def describe(objects):
    """Turn the list into a German sentence like 'Ich sehe: 2 Becher, Handy.'"""
    if not objects:
        return "Ich sehe nichts, was ich kenne."
    counts = Counter(o["german"] for o in objects)
    parts = [f"{n} {name}" if n > 1 else name for name, n in counts.items()]
    return "Ich sehe: " + ", ".join(parts) + "."


def live(index=0, min_confidence=0.4):
    """Show the camera feed with boxes around detected objects.
    Useful for testing and as an opener for the demo. Press q to quit."""
    import cv2

    model = load_model()
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {index}.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            result = model(frame, conf=min_confidence, verbose=False)[0]
            cv2.imshow("Object detection (q to quit)", result.plot())
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Object detection with YOLO")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--confidence", type=float, default=0.4,
                        help="Minimum confidence for a detection (0 to 1)")
    parser.add_argument("--live", action="store_true", help="Live view with boxes")
    parser.add_argument("--image", help="Use a saved image instead of the camera")
    parser.add_argument("--save", default="last_detection.jpg",
                        help="Where to save the frame with boxes (single-photo mode)")
    args = parser.parse_args()

    if args.live:
        live(args.camera, args.confidence)
        return

    if args.image:
        import cv2

        frame = cv2.imread(args.image)
        if frame is None:
            raise SystemExit(f"Could not read image {args.image}.")
    else:
        frame = capture_frame(args.camera)

    objects = detect(frame, args.confidence, save_path=args.save)
    print(f"Saved frame with boxes to {args.save}")
    for o in objects:
        print(f"  {o['name']:<15} {o['confidence']:.2f}  box {o['box']}")
    print(describe(objects))


if __name__ == "__main__":
    main()