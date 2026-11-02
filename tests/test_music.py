import time

from src.music_engine import MusicEngine


def main():

    music = MusicEngine()

    print("Playing C...")
    music.play_note("C")
    time.sleep(1)

    print("Playing E...")
    music.play_note("E")
    time.sleep(1)

    print("Playing G...")
    music.play_note("G")
    time.sleep(1)

    music.cleanup()

    print("Music test complete.")


if __name__ == "__main__":
    main()
