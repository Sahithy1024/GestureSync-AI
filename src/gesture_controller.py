import time


class GestureController:
    def __init__(self, music_engine):
        self.music = music_engine
        self.last_gesture = None
        self.last_action_time = 0
        self.cooldown = 0.5

    def handle_gesture(self, gesture):
        current_time = time.time()

        # Prevent the same gesture from triggering continuously
        if (
            gesture == self.last_gesture
            and current_time - self.last_action_time < self.cooldown
        ):
            return

        self.last_gesture = gesture
        self.last_action_time = current_time

        if gesture == "palm":
            print("Gesture: PALM -> C")
            self.music.play_note("C")

        elif gesture == "pinch":
            print("Gesture: PINCH -> E")
            self.music.play_note("E")

        elif gesture == "thumbs_up":
            print("Gesture: THUMBS UP -> G")
            self.music.play_note("G")

        elif gesture == "thumbs_down":
            print("Gesture: THUMBS DOWN -> A")
            self.music.play_note("A")

        elif gesture == "fist":
            print("Gesture: FIST -> STOP")
            self.music.stop_all()
