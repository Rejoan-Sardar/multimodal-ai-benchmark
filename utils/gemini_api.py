import os
import base64
import time
import google.generativeai as genai

class GeminiAPI:
    def __init__(self):
        # Get API key from environment
        api_key = os.getenv("GOOGLE_API_KEY", None)
        
        if not api_key:
            raise ValueError("Google API Key not found. Set the GOOGLE_API_KEY environment variable.")
        
        # Configure the Gemini API
        genai.configure(api_key=api_key)
        
        # Available models
        # Gemini 1.5 models represent the latest available models as of April, 2025
        self.text_models = [
            "gemini-pro", 
            "gemini-1.5-pro", 
            "gemini-1.5-flash",
            "gemini-1.5-pro-latest",
            "gemini-1.5-flash-latest"
        ]
        self.multimodal_models = [
            "gemini-pro-vision", 
            "gemini-1.5-pro-vision",
            "gemini-1.5-pro-vision-latest",
            "gemini-ultra-vision"
        ]
        
    def get_available_models(self):
        """Get list of available Gemini models"""
        return {
            "text": [{"id": model, "name": self._format_model_name(model)} for model in self.text_models],
            "multimodal": [{"id": model, "name": self._format_model_name(model)} for model in self.multimodal_models]
        }
    
    def _format_model_name(self, model_id):
        """Format model ID into a user-friendly name"""
        return model_id.replace("-", " ").title()
    
    def generate_text(self, model_id, prompt, temperature=0.7, max_output_tokens=1024):
        """Generate text using Gemini text models"""
        try:
            # Make sure we're using a valid text model
            if model_id not in self.text_models:
                if model_id in self.multimodal_models:
                    model_id = "gemini-pro"  # Fallback to text model
                else:
                    raise ValueError(f"Unsupported model: {model_id}")
            
            # Configure model
            model = genai.GenerativeModel(model_id)
            
            # Generate response
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                )
            )
            
            return {
                "text": response.text,
                "model_used": model_id
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "model_used": model_id
            }
    
    def generate_multimodal(self, model_id, prompt, image_data=None, temperature=0.7, max_output_tokens=1024):
        """Generate content using Gemini multimodal models"""
        try:
            # Make sure we're using a valid multimodal model
            if model_id not in self.multimodal_models:
                if model_id in self.text_models:
                    model_id = "gemini-pro-vision"  # Fallback to multimodal model
                else:
                    raise ValueError(f"Unsupported model: {model_id}")
            
            # Configure model
            model = genai.GenerativeModel(model_id)
            
            # Create content parts list
            content_parts = [prompt]
            
            # Add image if provided
            if image_data:
                if isinstance(image_data, str) and image_data.startswith("data:image"):
                    # Handle base64 image data
                    content_parts.append(image_data)
                elif isinstance(image_data, bytes):
                    # Handle raw image bytes
                    mime_type = "image/jpeg"  # Default assumption
                    content_parts.append({
                        "mime_type": mime_type,
                        "data": image_data
                    })
                elif isinstance(image_data, str) and os.path.exists(image_data):
                    # Handle image file path
                    with open(image_data, "rb") as file:
                        data = file.read()
                    mime_type = f"image/{os.path.splitext(image_data)[1][1:].lower()}"
                    content_parts.append({
                        "mime_type": mime_type,
                        "data": data
                    })
            
            # Generate response
            response = model.generate_content(
                content_parts,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                )
            )
            
            return {
                "text": response.text,
                "model_used": model_id
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "model_used": model_id
            }
    
    def analyze_video(self, model_id, video_path, prompt, frames_per_second=1):
        """Analyze video by extracting frames and sending to Gemini"""
        try:
            import cv2
            import numpy as np
            from PIL import Image
            from io import BytesIO
            
            # Make sure we're using a valid multimodal model
            if model_id not in self.multimodal_models:
                model_id = "gemini-pro-vision"  # Fallback to multimodal model
            
            # Configure model
            model = genai.GenerativeModel(model_id)
            
            # Open the video file
            video = cv2.VideoCapture(video_path)
            
            # Get basic video properties
            fps = video.get(cv2.CAP_PROP_FPS)
            frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps
            
            # Calculate frame extraction rate
            if frames_per_second > fps:
                frames_per_second = fps
                
            frame_interval = int(fps / frames_per_second)
            
            # Extract frames
            frames = []
            frame_number = 0
            
            while True:
                ret, frame = video.read()
                if not ret:
                    break
                
                if frame_number % frame_interval == 0:
                    # Convert frame to PIL Image
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(frame_rgb)
                    
                    # Convert to bytes
                    buffer = BytesIO()
                    pil_img.save(buffer, format="JPEG")
                    img_bytes = buffer.getvalue()
                    
                    # Encode as base64
                    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
                    img_data = f"data:image/jpeg;base64,{img_base64}"
                    
                    frames.append(img_data)
                
                frame_number += 1
                
                # Limit to a maximum of 20 frames to avoid token limits
                if len(frames) >= 20:
                    break
            
            # Release the video
            video.release()
            
            # Prepare content for Gemini
            content_parts = [prompt]
            
            # Add frames (up to 8 frames due to token limits)
            for img_data in frames[:8]:
                content_parts.append(img_data)
            
            # Generate response
            response = model.generate_content(
                content_parts,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=1024,
                )
            )
            
            return {
                "text": response.text,
                "model_used": model_id,
                "video_info": {
                    "duration": duration,
                    "fps": fps,
                    "frame_count": frame_count,
                    "frames_analyzed": len(frames)
                }
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "model_used": model_id
            }
    
    def analyze_audio(self, model_id, audio_path, prompt):
        """Analyze audio by transcribing and sending to Gemini"""
        try:
            import speech_recognition as sr
            
            # Make sure we're using a valid text model
            if model_id not in self.text_models:
                model_id = "gemini-pro"  # Fallback to text model
            
            # Configure model
            model = genai.GenerativeModel(model_id)
            
            # Initialize recognizer
            recognizer = sr.Recognizer()
            
            # Load audio file
            with sr.AudioFile(audio_path) as source:
                audio_data = recognizer.record(source)
                
                # Attempt transcription
                try:
                    transcription = recognizer.recognize_google(audio_data)
                except sr.UnknownValueError:
                    return {
                        "error": "Could not understand audio",
                        "model_used": model_id
                    }
                except sr.RequestError:
                    return {
                        "error": "Could not request results from Google Speech Recognition service",
                        "model_used": model_id
                    }
            
            # Combine prompt with transcription
            full_prompt = f"{prompt}\n\nAudio Transcription: {transcription}"
            
            # Generate response
            response = model.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=1024,
                )
            )
            
            return {
                "text": response.text,
                "transcription": transcription,
                "model_used": model_id
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "model_used": model_id
            }
