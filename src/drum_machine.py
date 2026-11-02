import os
import wave

import numpy as np
import pygame


class DrumMachine:
    def __init__(self):
        self.audio_dir = "audio"
        os.makedirs(self.audio_dir, exist_ok=True)

        self.sounds = {}

        self.create_drums()

    def create_sound(self, sound_type, filename, duration=0.3):
        sample_rate = 44100
        num_samples = int(sample_rate * duration)

        t = np.linspace(
            0,
            duration,
            num_samples,
            endpoint=False
        )

        # -----------------------------
        # Kick
        # -----------------------------

        if sound_type == "kick":
            frequency = 120 * np.exp(-8 * t)

            phase = (
                2
                * np.pi
                * np.cumsum(frequency)
                / sample_rate
            )

            audio = (
                np.sin(phase)
                * np.exp(-10 * t)
            )

        # -----------------------------
        # Snare
        # -----------------------------

        elif sound_type == "snare":
            noise = np.random.uniform(
                -1,
                1,
                num_samples
            )

            tone = np.sin(
                2 * np.pi * 180 * t
            )

            envelope = np.exp(-18 * t)

            audio = (
                0.8 * noise
                + 0.2 * tone
            ) * envelope

        # -----------------------------
        # Hi-hat
        # -----------------------------

        else:
            noise = np.random.uniform(
                -1,
                1,
                num_samples
            )

            envelope = np.exp(-35 * t)

            audio = noise * envelope

        # Normalize audio
        audio /= max(
            np.max(np.abs(audio)),
            1e-6
        )

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

    def create_drums(self):

        self.sounds["kick"] = self.create_sound(
            "kick",
            "kick.wav"
        )

        self.sounds["snare"] = self.create_sound(
            "snare",
            "snare.wav"
        )

        self.sounds["hihat"] = self.create_sound(
            "hihat",
            "hihat.wav"
        )

    # --------------------------------
    # Select drum based on hand position
    # --------------------------------

    def get_drum_from_position(
        self,
        x,
        frame_width
    ):

        third = frame_width / 3

        if x < third:
            return "kick"

        elif x < third * 2:
            return "snare"

        else:
            return "hihat"

    # --------------------------------
    # Play drum
    # --------------------------------

    def play(self, drum):

        if drum in self.sounds:

            sound = self.sounds[drum]

            sound.play()

    # --------------------------------
    # Cleanup
    # --------------------------------

    def cleanup(self):
        pass