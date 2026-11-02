import cv2

from src.hand_tracker import HandTracker
from src.feature_extractor import extract_landmarks
from src.predictor import GesturePredictor


def main():

    tracker = HandTracker(
        max_hands=1
    )

    predictor = GesturePredictor()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Gesture recognition started.")
    print("Press Q to quit.")

    while True:

        ret, frame = cap.read()

        if not ret:
            print("Failed to read webcam frame.")
            break

        # Mirror webcam
        frame = cv2.flip(
            frame,
            1
        )

        # Detect hand
        result = tracker.process(frame)

        # Draw landmarks
        frame = tracker.draw_landmarks(
            frame,
            result
        )

        gesture = "No hand"
        confidence = 0.0

        # If hand detected
        if result.hand_landmarks:

            hand = result.hand_landmarks[0]

            # Extract the same 63 features
            features = extract_landmarks(
                hand
            )

            # Neural network prediction
            gesture, confidence = predictor.predict(
                features
            )

        # Display prediction
        text = (
            f"Gesture: {gesture} "
            f"({confidence * 100:.1f}%)"
        )

        cv2.putText(
            frame,
            text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.imshow(
            "GestureSync AI",
            frame
        )

        # Quit with Q
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    tracker.close()


if __name__ == "__main__":
    main()
