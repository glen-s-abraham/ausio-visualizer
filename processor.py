import os
import numpy as np
import librosa
import cv2
from moviepy import VideoClip, AudioFileClip, ImageClip, CompositeVideoClip
from moviepy.video.fx import Resize
import yt_dlp
from PIL import Image, ImageFilter

def download_audio_from_youtube(url, output_path="temp_audio.mp3"):
    """Downloads audio from a YouTube URL."""
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': output_path.replace(".mp3", ""), # yt-dlp adds extension
        'quiet': True,
        'nocheckcertificate': True,
        'ignoreerrors': True,
        'no_warnings': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    
    # yt-dlp might append .mp3 to the filename we gave if it didn't have it, 
    # or if we gave it without extension it adds it. 
    # Let's ensure we return the correct path.
    expected_path = output_path
    if not output_path.endswith(".mp3"):
        expected_path += ".mp3"
    
    # Sometimes yt-dlp saves as .mp3 even if we didn't ask for it in the template explicitly if postprocessor is set
    # Check if file exists, if not check for .mp3 appended
    if not os.path.exists(expected_path):
        if os.path.exists(expected_path + ".mp3"):
            return expected_path + ".mp3"
    
    return expected_path

class AudioVisualizer:
    def __init__(self, audio_path, image_path, resolution=(1280, 720), fps=30, bar_color=(255, 255, 255), 
                 spectrum_opacity=0.8, spectrum_height_scale=0.3, smoothing_factor=5):
        self.audio_path = audio_path
        self.image_path = image_path
        self.resolution = resolution
        self.fps = fps
        self.bar_color = bar_color
        self.spectrum_opacity = spectrum_opacity
        self.spectrum_height_scale = spectrum_height_scale
        self.smoothing_factor = smoothing_factor
        
        # Load audio for analysis
        self.y, self.sr = librosa.load(audio_path, sr=None)
        self.duration = librosa.get_duration(y=self.y, sr=self.sr)
        
        # Analyze audio
        self.onset_env = librosa.onset.onset_strength(y=self.y, sr=self.sr)
        # Smooth onset envelope
        if self.smoothing_factor > 1:
            from scipy.ndimage import gaussian_filter1d
            self.onset_env = gaussian_filter1d(self.onset_env, sigma=self.smoothing_factor)
            
        self.times = librosa.times_like(self.onset_env, sr=self.sr)
        
        # Spectrum analysis
        self.n_fft = 2048
        self.hop_length = 512
        self.spec = np.abs(librosa.stft(self.y, n_fft=self.n_fft, hop_length=self.hop_length))
        self.spec_db = librosa.amplitude_to_db(self.spec, ref=np.max)
        
        # Smooth spectrum
        if self.smoothing_factor > 1:
            from scipy.ndimage import gaussian_filter1d
            # Apply smoothing across time axis
            self.spec_db = gaussian_filter1d(self.spec_db, sigma=self.smoothing_factor, axis=1)
        
        # Prepare images
        self.prepare_images()

    def prepare_images(self):
        # Load and process background
        img = Image.open(self.image_path).convert('RGB')
        
        # Create background (blurred and stretched)
        bg_img = img.resize(self.resolution)
        bg_img = bg_img.filter(ImageFilter.GaussianBlur(radius=30))
        self.bg_array = np.array(bg_img)
        
        # Create foreground (center artwork)
        # Determine size (e.g., 100% of height)
        target_h = int(self.resolution[1])
        aspect = img.width / img.height
        target_w = int(target_h * aspect)
        
        fg_img = img.resize((target_w, target_h), Image.LANCZOS)
        self.fg_array = np.array(fg_img)
        self.fg_center = (self.resolution[0] // 2, self.resolution[1] // 2)

    def get_audio_loudness(self, t):
        # Get current frame index
        frame_idx = int(t * self.sr / self.hop_length)
        if frame_idx >= len(self.times):
            frame_idx = len(self.times) - 1
        
        # Use onset strength for zoom effect
        return self.onset_env[frame_idx] if frame_idx < len(self.onset_env) else 0

    def get_spectrum_bars(self, t):
        frame_idx = int(t * self.sr / self.hop_length)
        if frame_idx >= self.spec_db.shape[1]:
            frame_idx = self.spec_db.shape[1] - 1
            
        # Get spectrum for this frame
        # We focus on lower frequencies for visual impact (first 100 bins)
        freqs = self.spec_db[:100, frame_idx]
        
        # Normalize to 0-1 range roughly
        # db is usually -80 to 0. Map -60 to 0 -> 0 to 1
        norm_freqs = (freqs + 60) / 60
        norm_freqs = np.clip(norm_freqs, 0, 1)
        
        return norm_freqs

    def make_frame(self, t):
        # 1. Pulsating Effect (Applied to Background)
        loudness = self.get_audio_loudness(t)
        zoom_factor = 1.0 + (loudness * 0.1) # Stronger zoom for background
        
        # To zoom in, we crop a smaller center region and resize up to resolution
        # crop_w = resolution_w / zoom_factor
        crop_w = int(self.resolution[0] / zoom_factor)
        crop_h = int(self.resolution[1] / zoom_factor)
        
        # Center of background
        bg_h, bg_w = self.bg_array.shape[:2]
        cx, cy = bg_w // 2, bg_h // 2
        
        x1 = max(0, cx - crop_w // 2)
        y1 = max(0, cy - crop_h // 2)
        x2 = min(bg_w, x1 + crop_w)
        y2 = min(bg_h, y1 + crop_h)
        
        # Crop and resize
        bg_crop = self.bg_array[y1:y2, x1:x2]
        frame = cv2.resize(bg_crop, self.resolution)
        
        # 2. Spectrum Effect
        bars = self.get_spectrum_bars(t)
        num_bars = len(bars)
        
        # Create a separate layer for spectrum to handle opacity
        spectrum_layer = frame.copy()
        
        # Draw bars at the bottom
        for i, val in enumerate(bars):
            height = int(val * (self.resolution[1] * self.spectrum_height_scale)) 
            
            # Calculate exact coordinates to cover full width
            x_start = int(i * self.resolution[0] / num_bars)
            x_end = int((i + 1) * self.resolution[0] / num_bars)
            
            y = self.resolution[1] - height
            
            # Draw rectangle
            cv2.rectangle(spectrum_layer, (x_start, y), (x_end, self.resolution[1]), self.bar_color, -1)
            
        # Blend spectrum layer with opacity
        cv2.addWeighted(spectrum_layer, self.spectrum_opacity, frame, 1 - self.spectrum_opacity, 0, frame)

        # 3. Static Foreground (Draw LAST so it's on top)
        h, w = self.fg_array.shape[:2]
        # Center it
        y1 = self.fg_center[1] - h // 2
        y2 = y1 + h
        x1 = self.fg_center[0] - w // 2
        x2 = x1 + w
        
        # Overlay foreground (simple copy since it's static size)
        # Handle boundaries just in case
        dy1, dy2 = max(0, y1), min(self.resolution[1], y2)
        dx1, dx2 = max(0, x1), min(self.resolution[0], x2)
        
        sy1 = max(0, dy1 - y1)
        sy2 = sy1 + (dy2 - dy1)
        sx1 = max(0, dx1 - x1)
        sx2 = sx1 + (dx2 - dx1)
        
        if dy2 > dy1 and dx2 > dx1:
            frame[dy1:dy2, dx1:dx2] = self.fg_array[sy1:sy2, sx1:sx2]
            
        return frame

    def create_video(self, output_path="output.mp4"):
        # Create video clip
        clip = VideoClip(self.make_frame, duration=self.duration)
        
        # Add audio
        audio = AudioFileClip(self.audio_path)
        clip = clip.with_audio(audio)
        
        # Write file
        clip.write_videofile(output_path, fps=self.fps, codec='libx264', audio_codec='aac')
        return output_path

if __name__ == "__main__":
    # Test run
    pass
