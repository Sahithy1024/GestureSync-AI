import os
import wave

import numpy as np
import pygame


class MusicEngine:

    def __init__(self):

        pygame.mixer.init(
            frequency=44100,
            size=-16,
            channels=1,
            buffer=512
        )

        self.volume = 0.5

        self.audio_dir = "audio"
        os.makedirs(self.audio_dir, exist_ok=True)

        self.sounds = {}

        self.create_notes()

        print("Music engine initialized.")

    def create_tone(self, frequency, filename, duration=0.4):

        sample_rate = 44100

        num_samples = int(
            sample_rate * duration
        )

        t = np.linspace(
            0,
            duration,
            num_samples,
            endpoint=False
        )

        # Fundamental + harmonics
        audio = (
            np.sin(2 * np.pi * frequency * t)
            + 0.30 * np.sin(2 * np.pi * frequency * 2 * t)
            + 0.15 * np.sin(2 * np.pi * frequency * 3 * t)
        )

        # Normalize
        audio /= np.max(np.abs(audio))

        # Convert to 16-bit PCM
        audio = (
            audio * 32767
        ).astype(np.int16)

        filepath = os.path.join(
            self.audio_dir,
            filename
        )

        with wave.open(filepath, "wb") as wav_file:

            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)

            wav_file.writeframes(
                audio.tobytes()
            )

        return pygame.mixer.Sound(filepath)

    def create_notes(self):

        notes = {
            "C": 261.63,
            "D": 293.66,
            "E": 329.63,
            "F": 349.23,
            "G": 392.00,
            "A": 440.00,
            "B": 493.88
        }

        for name, frequency in notes.items():

            filename = f"note_{name}.wav"

            self.sounds[name] = self.create_tone(
                frequency,
                filename
            )

    def play_note(self, note):

        if note not in self.sounds:
            return

        sound = self.sounds[note]

        sound.set_volume(
            self.volume
        )

        sound.play()

    def increase_volume(self):

        self.volume = min(
            1.0,
            self.volume + 0.1
        )

        print(
            f"Volume: {self.volume * 100:.0f}%"
        )

    def decrease_volume(self):

        self.volume = max(
            0.0,
            self.volume - 0.1
        )

        print(
            f"Volume: {self.volume * 100:.0f}%"
        )

    def stop_all(self):

        pygame.mixer.stop()

    def cleanup(self):

        pygame.mixer.stop()
        pygame.mixer.quit()
