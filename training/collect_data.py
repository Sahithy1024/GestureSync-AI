import csv
import os
import time

import cv2

from src.hand_tracker import HandTracker


DATA_FILE = "data/gestures.csv"

GESTURES = {
    ord("p"): "palm",
    ord("f"): "fist",
    ord("i"): "pinch",
    ord("u"): "thumbs_up",
    ord("d"): "thumbs_down",
}


def extract_landmarks(result):
    """Extract 63 normalized landmark features."""

    if not result.hand_landmarks:
        return None

    landmarks = result.hand_landmarks[0]

    wrist = landmarks[0]

    features = []

    for landmark in landmarks:
        features.extend([
            landmark.x - wrist.x,
            landmark.y - wrist.y,
            landmark.z - wrist.z,
        ])

    max_value = max(abs(value) for value in features)

    if max_value < 1e-6:
        return None

    return [value / max_value for value in features]


def create_dataset():
    os.makedirs("data", exist_ok=True)

    if os.path.exists(DATA_FILE):
        return

    with open(DATA_FILE, "w", newline="") as file:
        writer = csv.writer(file)

        header = [f"f{i}" for i in range(63)]
        header.append("label")

        writer.writerow(header)


def save_sample(features, label):
    with open(DATA_FILE, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(features + [label])


def main():

    # Start a fresh dataset.
    create_dataset()

    tracker = HandTracker(max_hands=1)

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("ERROR: Could not open webcam.")
        return

    current_label = None
    total_samples = 0

    print()
    print("=" * 55)
    print("GestureSync AI - Dataset Collector")
    print("=" * 55)
    print()
    print("P = Palm")
    print("F = Fist")
    print("I = Pinch")
    print("U = Thumbs Up")
    print("D = Thumbs Down")
    print("Q = Quit")
    print()
    print("After selecting a gesture, samples are collected")
    print("automatically every 0.3 seconds.")
    print()

    last_save_time = time.time()

    while True:

        success, frame = camera.read()

        if not success:
            print("ERROR: Could not read webcam.")
            break

        frame = cv2.flip(frame, 1)

        # Detect hand
        result = tracker.process(frame)

        # Draw hand
        frame = tracker.draw_landmarks(frame, result)

        # Extract features
        features = extract_landmarks(result)

        # Read keyboard
        key = cv2.waitKey(1) & 0xFF

        # Quit
        if key == ord("q"):
            break

        # Select gesture
        if key in GESTURES:

            current_label = GESTURES[key]

            print()
            print(
                f"Now collecting: {current_label}"
            )
            print(
                "Hold the gesture in front of the camera..."
            )

            # Reset timer for new gesture
            last_save_time = time.time()

        # Save automatically
        if (
            current_label is not None
            and features is not None
            and time.time() - last_save_time >= 0.3
        ):

            save_sample(
                features,
                current_label
            )

            total_samples += 1

            last_save_time = time.time()

            print(
                f"Saved #{total_samples} "
                f"({current_label})"
            )

        # UI
        cv2.putText(
            frame,
            "P Palm | F Fist | I Pinch | U Up | D Down | Q Quit",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )

        if current_label:
            cv2.putText(
                frame,
                f"Collecting: {current_label}",
                (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

        cv2.putText(
            frame,
            f"Total samples: {total_samples}",
            (10, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

        cv2.imshow(
            "GestureSync - Dataset Collector",
            frame
        )

    tracker.close()
    camera.release()
    cv2.destroyAllWindows()

    print()
    print("=" * 55)
    print("Collection finished.")
    print(f"Total samples: {total_samples}")
    print(f"Saved to: {DATA_FILE}")
    print("=" * 55)


if __name__ == "__main__":
    main()
