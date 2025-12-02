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
                 spectrum_opacity=0.8, spectrum_height_scale=0.3, smoothing_factor=5, 
                 template="linear", num_bars=60, logo_path=None, blur_radius=30):
        self.audio_path = audio_path
        self.image_path = image_path
        self.resolution = resolution
        self.fps = fps
        self.bar_color = bar_color
        self.spectrum_opacity = spectrum_opacity
        self.spectrum_height_scale = spectrum_height_scale
        self.smoothing_factor = smoothing_factor
        self.template = template
        self.num_bars = num_bars
        self.logo_path = logo_path
        self.blur_radius = blur_radius
        
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
        bg_img = bg_img.filter(ImageFilter.GaussianBlur(radius=self.blur_radius))
        self.bg_array = np.array(bg_img)
        
        # Create foreground (center artwork)
        # For Linear: Full height
        target_h_linear = int(self.resolution[1])
        aspect = img.width / img.height
        target_w_linear = int(target_h_linear * aspect)
        fg_img_linear = img.resize((target_w_linear, target_h_linear), Image.LANCZOS)
        self.fg_array_linear = np.array(fg_img_linear)
        
        # For Circular: Square crop then circle mask
        # Determine size (e.g., 40% of height for the circle diameter)
        circle_diam = int(self.resolution[1] * 0.4)
        
        if self.logo_path:
            # Use provided logo
            img_to_circle = Image.open(self.logo_path).convert('RGB')
        else:
            # Use main image
            img_to_circle = img
            
        # Crop center square from image
        min_dim = min(img_to_circle.width, img_to_circle.height)
        left = (img_to_circle.width - min_dim) // 2
        top = (img_to_circle.height - min_dim) // 2
        img_square = img_to_circle.crop((left, top, left + min_dim, top + min_dim))
        img_square = img_square.resize((circle_diam, circle_diam), Image.LANCZOS)
        
        # Create circular mask
        mask = Image.new('L', (circle_diam, circle_diam), 0)
        from PIL import ImageDraw
        draw = ImageDraw.Draw(mask)
        draw.ellipse((0, 0, circle_diam, circle_diam), fill=255)
        
        # Apply mask
        img_circular = Image.new('RGBA', (circle_diam, circle_diam), (0, 0, 0, 0))
        img_circular.paste(img_square, (0, 0), mask=mask)
        
        # Convert to numpy array (keep alpha for transparency)
        self.fg_array_circular = np.array(img_circular)
        self.fg_center = (self.resolution[0] // 2, self.resolution[1] // 2)

    def get_audio_loudness(self, t):
        # Get current frame index
        frame_idx = int(t * self.sr / self.hop_length)
        if frame_idx >= len(self.times):
            frame_idx = len(self.times) - 1
        
        # Use onset strength for zoom effect
        return self.onset_env[frame_idx] if frame_idx < len(self.onset_env) else 0

    def get_spectrum_bars(self, t, num_bars_override=None):
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
        
        # Resample to desired number of bars
        target_bars = num_bars_override if num_bars_override else 100 # Default for linear
        if len(norm_freqs) != target_bars:
            # Simple resampling using interpolation
            x_old = np.linspace(0, 1, len(norm_freqs))
            x_new = np.linspace(0, 1, target_bars)
            norm_freqs = np.interp(x_new, x_old, norm_freqs)
            
        return norm_freqs

    def render_linear_frame(self, frame, t):
        # 2. Spectrum Effect
        bars = self.get_spectrum_bars(t, num_bars_override=100)
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
            
            # Draw rectangle (Bottom)
            cv2.rectangle(spectrum_layer, (x_start, y), (x_end, self.resolution[1]), self.bar_color, -1)
            
            # Draw rectangle (Top - Inverted & Mirrored)
            # Use the value from the opposite end of the spectrum array
            val_top = bars[num_bars - 1 - i]
            height_top = int(val_top * (self.resolution[1] * self.spectrum_height_scale))
            cv2.rectangle(spectrum_layer, (x_start, 0), (x_end, height_top), self.bar_color, -1)
            
        # Blend spectrum layer with opacity
        cv2.addWeighted(spectrum_layer, self.spectrum_opacity, frame, 1 - self.spectrum_opacity, 0, frame)

        # 3. Static Foreground (Draw LAST so it's on top)
        h, w = self.fg_array_linear.shape[:2]
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
            frame[dy1:dy2, dx1:dx2] = self.fg_array_linear[sy1:sy2, sx1:sx2]
            
        return frame

    def render_circular_frame(self, frame, t):
        import math
        
        # 2. Circular Spectrum
        bars = self.get_spectrum_bars(t, num_bars_override=self.num_bars)
        num_bars = len(bars)
        
        spectrum_layer = frame.copy()
        
        cx, cy = self.fg_center
        radius = self.fg_array_circular.shape[0] // 2 + 10 # Start slightly outside the logo
        max_bar_height = int(self.resolution[1] * self.spectrum_height_scale)
        
        for i, val in enumerate(bars):
            bar_height = int(val * max_bar_height)
            angle_deg = (i / num_bars) * 360 - 90 # Start from top
            angle_rad = math.radians(angle_deg)
            
            # Start point (on circle edge)
            x1 = int(cx + radius * math.cos(angle_rad))
            y1 = int(cy + radius * math.sin(angle_rad))
            
            # End point (outwards)
            x2 = int(cx + (radius + bar_height) * math.cos(angle_rad))
            y2 = int(cy + (radius + bar_height) * math.sin(angle_rad))
            
            # Draw line (thickness proportional to circumference / num_bars)
            thickness = max(2, int((2 * math.pi * radius) / num_bars) - 2)
            cv2.line(spectrum_layer, (x1, y1), (x2, y2), self.bar_color, thickness)
            
        # Blend spectrum
        cv2.addWeighted(spectrum_layer, self.spectrum_opacity, frame, 1 - self.spectrum_opacity, 0, frame)
        
        # 3. Circular Logo
        # Overlay with alpha channel
        fg_h, fg_w = self.fg_array_circular.shape[:2]
        y1 = cy - fg_h // 2
        y2 = y1 + fg_h
        x1 = cx - fg_w // 2
        x2 = x1 + fg_w
        
        # Alpha blending
        alpha_s = self.fg_array_circular[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s
        
        # Region of interest
        roi = frame[y1:y2, x1:x2]
        
        for c in range(0, 3):
            roi[:, :, c] = (alpha_s * self.fg_array_circular[:, :, c] +
                            alpha_l * roi[:, :, c])
                            
        frame[y1:y2, x1:x2] = roi
        
        return frame

    def make_frame(self, t):
        # 1. Pulsating Background (Common)
        loudness = self.get_audio_loudness(t)
        zoom_factor = 1.0 + (loudness * 0.1) 
        
        crop_w = int(self.resolution[0] / zoom_factor)
        crop_h = int(self.resolution[1] / zoom_factor)
        
        bg_h, bg_w = self.bg_array.shape[:2]
        cx, cy = bg_w // 2, bg_h // 2
        
        x1 = max(0, cx - crop_w // 2)
        y1 = max(0, cy - crop_h // 2)
        x2 = min(bg_w, x1 + crop_w)
        y2 = min(bg_h, y1 + crop_h)
        
        bg_crop = self.bg_array[y1:y2, x1:x2]
        frame = cv2.resize(bg_crop, self.resolution)
        
        # Dispatch
        if self.template == "circular":
            return self.render_circular_frame(frame, t)
        else:
            return self.render_linear_frame(frame, t)

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
