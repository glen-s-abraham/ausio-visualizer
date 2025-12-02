import streamlit as st
import os
import tempfile
from processor import AudioVisualizer, download_audio_from_youtube
from PIL import Image

st.set_page_config(page_title="Audio Visualizer", layout="wide")

st.title("🎵 Audio Visualizer Generator")
st.markdown("Create professional music videos with spectrum visualization.")

# Sidebar for configuration
st.sidebar.header("Configuration")
bar_color_hex = st.sidebar.color_picker("Spectrum Color", "#00FF00")
# Convert hex to RGB
bar_color = tuple(int(bar_color_hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
# OpenCV uses BGR, so reverse it
bar_color = bar_color[::-1]

fps = st.sidebar.slider("FPS", 15, 60, 30)
resolution_options = {"720p": (1280, 720), "1080p": (1920, 1080)}
res_name = st.sidebar.selectbox("Resolution", list(resolution_options.keys()))
resolution = resolution_options[res_name]

st.sidebar.subheader("Visual Settings")
spectrum_opacity = st.sidebar.slider("Spectrum Opacity", 0.1, 1.0, 0.8)
spectrum_height = st.sidebar.slider("Spectrum Height", 0.1, 0.8, 0.3)
smoothing = st.sidebar.slider("Motion Smoothing", 1, 20, 5)

# Main Area
col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Audio Source")
    audio_source = st.radio("Choose Source", ["Upload File", "YouTube URL"])
    
    audio_path = None
    if audio_source == "Upload File":
        uploaded_audio = st.file_uploader("Upload MP3/WAV", type=["mp3", "wav"])
        if uploaded_audio:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_audio:
                tmp_audio.write(uploaded_audio.read())
                audio_path = tmp_audio.name
    else:
        yt_url = st.text_input("YouTube URL")
        if yt_url:
            if st.button("Download Audio"):
                with st.spinner("Downloading audio..."):
                    try:
                        # Use a temp directory
                        temp_dir = tempfile.gettempdir()
                        output_path = os.path.join(temp_dir, "yt_audio.mp3")
                        audio_path = download_audio_from_youtube(yt_url, output_path)
                        st.success(f"Downloaded: {os.path.basename(audio_path)}")
                        # Store in session state to persist
                        st.session_state['audio_path'] = audio_path
                    except Exception as e:
                        st.error(f"Error downloading: {e}")
            
            if 'audio_path' in st.session_state:
                audio_path = st.session_state['audio_path']

with col2:
    st.subheader("2. Artwork")
    uploaded_image = st.file_uploader("Upload Artwork (JPG/PNG)", type=["jpg", "jpeg", "png"])
    image_path = None
    if uploaded_image:
        # Show preview
        st.image(uploaded_image, caption="Artwork Preview", width=300)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_img:
            tmp_img.write(uploaded_image.read())
            image_path = tmp_img.name

st.divider()

if st.button("Generate Video", type="primary"):
    if not audio_path or not image_path:
        st.error("Please provide both audio and artwork.")
    else:
        st.info("Generating video... This may take a while.")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            output_file = "output_video.mp4"
            
            # Initialize visualizer
            status_text.text("Analyzing audio...")
            viz = AudioVisualizer(audio_path, image_path, resolution=resolution, fps=fps, bar_color=bar_color,
                                  spectrum_opacity=spectrum_opacity, spectrum_height_scale=spectrum_height, smoothing_factor=smoothing)
            
            # Generate
            status_text.text("Rendering video frames...")
            # We can't easily hook into moviepy progress bar from here without custom logger, 
            # so we'll just show a spinner for the main render.
            with st.spinner("Rendering..."):
                viz.create_video(output_file)
            
            st.success("Video generated successfully!")
            st.video(output_file)
            
            with open(output_file, "rb") as f:
                st.download_button("Download Video", f, file_name="visualizer.mp4")
                
        except Exception as e:
            st.error(f"An error occurred: {e}")
            import traceback
            st.code(traceback.format_exc())

