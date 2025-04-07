import os
import tempfile
import base64
from io import BytesIO
from PIL import Image
import librosa
import soundfile as sf
import numpy as np
import cv2
import ffmpeg

def process_file(uploaded_file, file_type):
    """
    Process uploaded files based on their type
    
    Args:
        uploaded_file: The Streamlit uploaded file object
        file_type: The type of file (text, image, video, audio)
        
    Returns:
        dict: Processed data and metadata
    """
    if not uploaded_file:
        return None
    
    try:
        # Handle different file types
        if file_type == "text":
            return process_text_file(uploaded_file)
        elif file_type == "image":
            return process_image_file(uploaded_file)
        elif file_type == "video":
            return process_video_file(uploaded_file)
        elif file_type == "audio":
            return process_audio_file(uploaded_file)
        else:
            return {"error": f"Unsupported file type: {file_type}"}
    
    except Exception as e:
        return {"error": f"Error processing file: {str(e)}"}

def process_text_file(uploaded_file):
    """Process text files"""
    try:
        # Read file content as string
        content = uploaded_file.getvalue().decode("utf-8")
        
        return {
            "content": content,
            "size": len(content),
            "filename": uploaded_file.name,
            "file_type": "text"
        }
    except Exception as e:
        return {"error": f"Error processing text file: {str(e)}"}

def process_image_file(uploaded_file):
    """Process image files"""
    try:
        # Open image using PIL
        image = Image.open(BytesIO(uploaded_file.getvalue()))
        
        # Get image properties
        width, height = image.size
        format = image.format
        mode = image.mode
        
        # Convert to base64 for API calls
        buffered = BytesIO()
        image.save(buffered, format=format)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return {
            "image": image,
            "base64": f"data:image/{format.lower()};base64,{img_str}",
            "width": width,
            "height": height,
            "format": format,
            "mode": mode,
            "filename": uploaded_file.name,
            "file_type": "image"
        }
    except Exception as e:
        return {"error": f"Error processing image file: {str(e)}"}

def process_video_file(uploaded_file):
    """Process video files"""
    try:
        # Save file to temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_path = tmp_file.name
        
        # Get video properties using OpenCV
        video = cv2.VideoCapture(temp_path)
        
        if not video.isOpened():
            os.unlink(temp_path)
            return {"error": "Could not open video file"}
        
        # Get video metadata
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        
        # Extract first frame as thumbnail
        ret, frame = video.read()
        thumbnail = None
        
        if ret:
            # Convert to RGB (from BGR)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            thumbnail = Image.fromarray(frame_rgb)
            
            # Convert thumbnail to base64
            buffered = BytesIO()
            thumbnail.save(buffered, format="JPEG")
            thumbnail_str = base64.b64encode(buffered.getvalue()).decode()
            thumbnail_base64 = f"data:image/jpeg;base64,{thumbnail_str}"
        else:
            thumbnail_base64 = None
        
        # Release video
        video.release()
        
        return {
            "path": temp_path,
            "fps": fps,
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "duration": duration,
            "thumbnail": thumbnail_base64,
            "filename": uploaded_file.name,
            "file_type": "video",
            "temp_file": True  # Flag to indicate temporary file needs cleanup
        }
    except Exception as e:
        # Clean up temp file if exists
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.unlink(temp_path)
        return {"error": f"Error processing video file: {str(e)}"}

def process_audio_file(uploaded_file):
    """Process audio files"""
    try:
        # Save file to temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_path = tmp_file.name
        
        # Load audio using librosa
        y, sr = librosa.load(temp_path, sr=None)
        
        # Get audio properties
        duration = librosa.get_duration(y=y, sr=sr)
        
        # Generate waveform visualization data
        # Use a downsampled version to keep the data size reasonable
        if len(y) > 10000:
            step = len(y) // 10000
            waveform_data = y[::step].tolist()[:10000]
        else:
            waveform_data = y.tolist()
        
        return {
            "path": temp_path,
            "sample_rate": sr,
            "duration": duration,
            "channels": 1 if y.ndim == 1 else y.shape[1],
            "waveform_data": waveform_data,
            "filename": uploaded_file.name,
            "file_type": "audio",
            "temp_file": True  # Flag to indicate temporary file needs cleanup
        }
    except Exception as e:
        # Clean up temp file if exists
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.unlink(temp_path)
        return {"error": f"Error processing audio file: {str(e)}"}

def cleanup_temp_files(file_data):
    """Clean up temporary files created during processing"""
    if file_data and file_data.get("temp_file") and "path" in file_data:
        try:
            if os.path.exists(file_data["path"]):
                os.unlink(file_data["path"])
        except Exception as e:
            print(f"Error cleaning up temporary file: {e}")
