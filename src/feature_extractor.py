import numpy as np


def extract_landmarks(hand_landmarks):
    """
    Convert 21 MediaPipe hand landmarks
    into 63 normalized features.
    """

    features = []

    # Wrist coordinates
    wrist_x = hand_landmarks[0].x
    wrist_y = hand_landmarks[0].y
    wrist_z = hand_landmarks[0].z

    # Make coordinates relative to wrist
    for landmark in hand_landmarks:

        x = landmark.x - wrist_x
        y = landmark.y - wrist_y
        z = landmark.z - wrist_z

        features.extend([x, y, z])

    features = np.array(
        features,
        dtype=np.float32
    )

    # Normalize
    max_value = np.max(
        np.abs(features)
    )

    if max_value > 0:
        features = features / max_value

    return features
