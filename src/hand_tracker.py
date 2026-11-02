import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class HandTracker:
    def __init__(self, max_hands=2):
        self.max_hands = max_hands

        # Create MediaPipe Hand Landmarker options
        base_options = python.BaseOptions(
            model_asset_path="../models/hand_landmarker.task"
        )

        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.detector = vision.HandLandmarker.create_from_options(options)

        self.timestamp_ms = 0

    def process(self, frame):
        """
        Detect hands in an OpenCV BGR frame.

        Returns:
            MediaPipe HandLandmarkerResult
        """

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert OpenCV image to MediaPipe image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        self.timestamp_ms += 33

        result = self.detector.detect_for_video(
            mp_image,
            self.timestamp_ms
        )

        return result

    def draw_landmarks(self, frame, result):
        """
        Draw detected hand landmarks.
        """

        if result.hand_landmarks:
            for hand_landmarks in result.hand_landmarks:

                # Draw landmark points
                for landmark in hand_landmarks:
                    h, w, _ = frame.shape

                    x = int(landmark.x * w)
                    y = int(landmark.y * h)

                    cv2.circle(
                        frame,
                        (x, y),
                        5,
                        (0, 255, 0),
                        -1
                    )

                # Draw connections
                connections = [
                    (0, 1), (1, 2), (2, 3), (3, 4),
                    (0, 5), (5, 6), (6, 7), (7, 8),
                    (5, 9), (9, 10), (10, 11), (11, 12),
                    (9, 13), (13, 14), (14, 15), (15, 16),
                    (13, 17), (17, 18), (18, 19), (19, 20),
                    (0, 17)
                ]

                h, w, _ = frame.shape

                for start, end in connections:
                    x1 = int(hand_landmarks[start].x * w)
                    y1 = int(hand_landmarks[start].y * h)

                    x2 = int(hand_landmarks[end].x * w)
                    y2 = int(hand_landmarks[end].y * h)

                    cv2.line(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

        return frame

    def close(self):
        self.detector.close()
