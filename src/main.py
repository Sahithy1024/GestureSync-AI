import cv2

from drum_machine import DrumMachine
from virtual_piano import VirtualPiano
from hand_tracker import HandTracker
from feature_extractor import extract_landmarks
from predictor import GesturePredictor
from music_engine import MusicEngine
from gesture_controller import GestureController


def main():
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("ERROR: Could not open webcam.")
        return

    tracker = HandTracker(max_hands=1)
    predictor = GesturePredictor()

    music = MusicEngine()
    drums = DrumMachine()

    controller = GestureController(music)
    piano = VirtualPiano()

    previous_gesture = "No hand"

    print("\nGestureMusicStudio")
    print("==================")
    print("Webcam started.")
    print()
    print("Palm        -> C")
    print("Pinch       -> Piano + Drum")
    print("Thumbs Up   -> G")
    print("Thumbs Down -> A")
    print("Fist        -> Stop")
    print()
    print("Press Q to quit.\n")

    while True:
        success, frame = camera.read()

        if not success:
            print("ERROR: Could not read frame.")
            break

        frame = cv2.flip(frame, 1)

        results = tracker.process(frame)
        frame = tracker.draw_landmarks(frame, results)

        gesture = "No hand"
        confidence = 0.0
        selected_note = "-"
        selected_drum = "-"

        height, width, _ = frame.shape

        # --------------------------------
        # Draw drum zones
        # --------------------------------

        drum_y1 = 175
        drum_y2 = 270

        third = width // 3

        cv2.rectangle(
            frame,
            (0, drum_y1),
            (third, drum_y2),
            (60, 60, 60),
            -1
        )

        cv2.rectangle(
            frame,
            (third, drum_y1),
            (third * 2, drum_y2),
            (90, 90, 90),
            -1
        )

        cv2.rectangle(
            frame,
            (third * 2, drum_y1),
            (width, drum_y2),
            (120, 120, 120),
            -1
        )

        cv2.putText(
            frame,
            "KICK",
            (third // 2 - 35, 225),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "SNARE",
            (third + third // 2 - 45, 225),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "HI-HAT",
            (third * 2 + third // 2 - 45, 225),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        # --------------------------------
        # Hand detected
        # --------------------------------

        if results.hand_landmarks:

            hand_landmarks = results.hand_landmarks[0]

            features = extract_landmarks(hand_landmarks)

            gesture, confidence = predictor.predict(features)

            hand_x = int(
                hand_landmarks[9].x * width
            )

            # Piano note
            selected_note = piano.get_note_from_position(
                hand_x,
                width
            )

            # Drum
            selected_drum = drums.get_drum_from_position(
                hand_x,
                width
            )

            # Pinch triggers both
            if (
                gesture == "pinch"
                and previous_gesture != "pinch"
            ):
                music.play_note(selected_note)
                drums.play(selected_drum)

                print(
                    f"Played piano: {selected_note} | "
                    f"Drum: {selected_drum}"
                )

            # Other gestures
            if gesture != "pinch":
                controller.handle_gesture(gesture)

            previous_gesture = gesture

            # Highlight current drum zone
            if selected_drum == "kick":
                cv2.rectangle(
                    frame,
                    (0, drum_y1),
                    (third, drum_y2),
                    (255, 255, 255),
                    3
                )

            elif selected_drum == "snare":
                cv2.rectangle(
                    frame,
                    (third, drum_y1),
                    (third * 2, drum_y2),
                    (255, 255, 255),
                    3
                )

            elif selected_drum == "hihat":
                cv2.rectangle(
                    frame,
                    (third * 2, drum_y1),
                    (width, drum_y2),
                    (255, 255, 255),
                    3
                )

            # Selected note
            cv2.putText(
                frame,
                f"Note: {selected_note}",
                (20, 310),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

            # Selected drum
            cv2.putText(
                frame,
                f"Drum: {selected_drum}",
                (180, 310),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 200, 0),
                2
            )

        else:
            previous_gesture = "No hand"

        # --------------------------------
        # Prediction display
        # --------------------------------

        cv2.rectangle(
            frame,
            (10, 10),
            (430, 100),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            frame,
            f"Gesture: {gesture}",
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Confidence: {confidence * 100:.1f}%",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        # --------------------------------
        # Virtual piano
        # --------------------------------

        piano.draw(frame)

        # --------------------------------
        # Show
        # --------------------------------

        cv2.imshow(
            "GestureMusicStudio - AI Music Control",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    tracker.close()
    music.cleanup()

    camera.release()
    cv2.destroyAllWindows()

    print("GestureMusicStudio stopped.")


if __name__ == "__main__":
    main()
