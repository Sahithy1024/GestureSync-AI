import cv2


class VirtualPiano:
    def __init__(self):
        self.notes = ["C", "D", "E", "F", "G", "A", "B"]

    def get_note_from_position(self, x, frame_width):
        zone_width = frame_width / len(self.notes)

        index = int(x / zone_width)

        # Keep index inside valid range
        index = max(0, min(index, len(self.notes) - 1))

        return self.notes[index]

    def draw(self, frame):
        height, width, _ = frame.shape
        zone_width = width // len(self.notes)

        for i, note in enumerate(self.notes):
            x1 = i * zone_width
            x2 = (i + 1) * zone_width

            # Draw piano zone
            cv2.rectangle(
                frame,
                (x1, height - 150),
                (x2, height),
                (255, 255, 255),
                -1
            )

            cv2.rectangle(
                frame,
                (x1, height - 150),
                (x2, height),
                (0, 0, 0),
                2
            )

            # Note label
            cv2.putText(
                frame,
                note,
                (x1 + zone_width // 2 - 10, height - 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 0, 0),
                2
            )

        return frame
