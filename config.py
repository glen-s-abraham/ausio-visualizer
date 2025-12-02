import os

# Default configuration
BASE_OUTPUT_DIR = r"/home/glen-personal/Videos/av-out"

# Create directory if it doesn't exist
if not os.path.exists(BASE_OUTPUT_DIR):
    os.makedirs(BASE_OUTPUT_DIR)

AUDIO_DIR = os.path.join(BASE_OUTPUT_DIR, "audio")
IMAGE_DIR = os.path.join(BASE_OUTPUT_DIR, "images")
VIDEO_DIR = os.path.join(BASE_OUTPUT_DIR, "videos")

for d in [AUDIO_DIR, IMAGE_DIR, VIDEO_DIR]:
    if not os.path.exists(d):
        os.makedirs(d)
