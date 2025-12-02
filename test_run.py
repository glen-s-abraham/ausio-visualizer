import numpy as np
import soundfile as sf
from PIL import Image
from processor import AudioVisualizer
import os

def create_dummy_assets():
    # Create 5 seconds of white noise audio
    sr = 22050
    duration = 5
    y = np.random.uniform(-0.5, 0.5, int(sr * duration))
    sf.write('test_audio.wav', y, sr)
    
    # Create a dummy image
    img = Image.new('RGB', (500, 500), color = 'red')
    img.save('test_image.png')
    
    return 'test_audio.wav', 'test_image.png'

def test_visualizer():
    print("Creating dummy assets...")
    audio_path, image_path = create_dummy_assets()
    
    print("Initializing Visualizer (Circular with Logo)...")
    # Use image_path as logo_path for testing
    viz = AudioVisualizer(audio_path, image_path, resolution=(640, 360), fps=10, 
                          spectrum_opacity=0.5, spectrum_height_scale=0.5, smoothing_factor=10,
                          template="circular", num_bars=30, logo_path=image_path, blur_radius=50)
    
    print("Generating Video...")
    output_path = "test_output.mp4"
    viz.create_video(output_path)
    
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        print("SUCCESS: Video generated successfully.")
    else:
        print("FAILURE: Video file not found or empty.")

if __name__ == "__main__":
    test_visualizer()
