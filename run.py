import cv2
from pianoputer.pianoputer import *
from finger_tracker.finger_tracker import finger_tracker

parser = get_parser()
wav_path, keyboard_path, clear_cache = process_args(parser, None)
audio_data, framerate_hz, channels = get_audio_data(wav_path)

keys = ["1", "2", "3", "4", "5", "6", "7", "8"]
tones = [0, 2, 4, 5, 7, 9, 11, 12]
pygame.mixer.init(
    framerate_hz,
    BITS_32BIT,
    channels,
    allowedchanges=AUDIO_ALLOWED_CHANGES_HARDWARE_DETERMINED,
)
key_sounds = get_or_create_key_sounds(wav_path, framerate_hz, channels, tones, clear_cache, keys=keys)
sound_by_key = dict(zip(keys, key_sounds))

cap = cv2.VideoCapture(0)
success, img = cap.read()
ft = finger_tracker(img)
ft.set_sound(sound_by_key)

while True:
    success, img = cap.read()

    if success:
        img = ft.process(img)
    
    cv2.imshow('', img)
    cv2.waitKey(10)