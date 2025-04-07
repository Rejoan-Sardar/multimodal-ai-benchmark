import os
import base64
import time
import json
from openai import OpenAI

class OpenAIAPI:
    def __init__(self):
        # Get API key from environment
        api_key = os.getenv("OPENAI_API_KEY", None)
        
        if not api_key:
            raise ValueError("OpenAI API Key not found. Set the OPENAI_API_KEY environment variable.")
        
        # Initialize the OpenAI client
        self.client = OpenAI(api_key=api_key)
        
        # Available models
        # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
        # do not change this unless explicitly requested by the user
        self.text_models = [
            "gpt-4o", 
            "gpt-4-turbo", 
            "gpt-4o-mini",
            "gpt-4",
            "gpt-3.5-turbo"
        ]
        self.multimodal_models = [
            "gpt-4o", 
            "gpt-4-vision", 
            "gpt-4o-vision",
            "gpt-4-turbo-vision"
        ]
        self.image_models = ["dall-e-3", "dall-e-2"]
        
    def get_available_models(self):
        """Get list of available OpenAI models"""
        return {
            "text": [{"id": model, "name": self._format_model_name(model)} for model in self.text_models],
            "multimodal": [{"id": model, "name": self._format_model_name(model)} for model in self.multimodal_models],
            "image": [{"id": model, "name": self._format_model_name(model)} for model in self.image_models]
        }
    
    def _format_model_name(self, model_id):
        """Format model ID into a user-friendly name"""
        return model_id.replace("-", " ").upper()
    
    def generate_text(self, model_id, prompt, temperature=0.7, max_tokens=1024):
        """Generate text using OpenAI text models"""
        try:
            # Make sure we're using a valid text model
            if model_id not in self.text_models:
                if model_id in self.multimodal_models:
                    model_id = "gpt-4o"  # Default to newest model
                else:
                    raise ValueError(f"Unsupported model: {model_id}")
            
            # Generate response
            response = self.client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return {
                "text": response.choices[0].message.content,
                "model_used": model_id
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "model_used": model_id
            }
    
    def generate_multimodal(self, model_id, prompt, image_data=None, temperature=0.7, max_tokens=1024):
        """Generate content using OpenAI multimodal models"""
        try:
            # Make sure we're using a valid multimodal model
            if model_id not in self.multimodal_models:
                model_id = "gpt-4o"  # Default to newest model
            
            # Create messages list
            if image_data:
                # Handle different image data formats
                if isinstance(image_data, str) and image_data.startswith("data:image"):
                    # Base64 image data
                    image_url = image_data
                elif isinstance(image_data, bytes):
                    # Raw image bytes
                    image_base64 = base64.b64encode(image_data).decode('utf-8')
                    image_url = f"data:image/jpeg;base64,{image_base64}"
                elif isinstance(image_data, str) and os.path.exists(image_data):
                    # Image file path
                    with open(image_data, "rb") as file:
                        image_bytes = file.read()
                    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                    image_url = f"data:image/jpeg;base64,{image_base64}"
                else:
                    raise ValueError("Unsupported image data format")
                
                # Create multimodal message
                messages = [
                    {"role": "system", "content": "You are a helpful AI assistant."},
                    {
                        "role": "user", 
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": image_url}}
                        ]
                    }
                ]
            else:
                # Text-only message
                messages = [
                    {"role": "system", "content": "You are a helpful AI assistant."},
                    {"role": "user", "content": prompt}
                ]
            
            # Generate response
            response = self.client.chat.completions.create(
                model=model_id,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return {
                "text": response.choices[0].message.content,
                "model_used": model_id
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "model_used": model_id
            }
    
    def analyze_video(self, model_id, video_path, prompt, frames_per_second=1):
        """Analyze video by extracting frames and sending to OpenAI"""
        try:
            import cv2
            import numpy as np
            from PIL import Image
            from io import BytesIO
            
            # Make sure we're using a valid multimodal model
            if model_id not in self.multimodal_models:
                model_id = "gpt-4o"  # Default to newest model
            
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
                    
                    # Convert to base64
                    buffer = BytesIO()
                    pil_img.save(buffer, format="JPEG")
                    img_bytes = buffer.getvalue()
                    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
                    
                    frames.append(f"data:image/jpeg;base64,{img_base64}")
                
                frame_number += 1
                
                # Limit to a maximum of 20 frames
                if len(frames) >= 20:
                    break
            
            # Release the video
            video.release()
            
            # Prepare content for OpenAI (limited to 5 frames due to token constraints)
            content = [
                {"type": "text", "text": f"{prompt}\n\nAnalyze the following video frames:"}
            ]
            
            # Add frames (limit to 5 to avoid token limits)
            for img_data in frames[:5]:
                content.append(
                    {"type": "image_url", "image_url": {"url": img_data}}
                )
            
            # Generate response
            response = self.client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant."},
                    {"role": "user", "content": content}
                ],
                temperature=0.7,
                max_tokens=1024
            )
            
            return {
                "text": response.choices[0].message.content,
                "model_used": model_id,
                "video_info": {
                    "duration": duration,
                    "fps": fps,
                    "frame_count": frame_count,
                    "frames_analyzed": len(frames[:5])
                }
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "model_used": model_id
            }
    
    def analyze_audio(self, model_id, audio_path, prompt):
        """Analyze audio by transcribing with Whisper and sending to OpenAI"""
        try:
            # Make sure we're using a valid text model
            if model_id not in self.text_models:
                model_id = "gpt-4o"  # Default to newest model
            
            # Transcribe using Whisper
            with open(audio_path, "rb") as audio_file:
                transcription_response = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )
                
            transcription = transcription_response.text
            
            # Combine prompt with transcription
            full_prompt = f"{prompt}\n\nAudio Transcription: {transcription}"
            
            # Generate response
            response = self.client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant."},
                    {"role": "user", "content": full_prompt}
                ],
                temperature=0.7,
                max_tokens=1024
            )
            
            return {
                "text": response.choices[0].message.content,
                "transcription": transcription,
                "model_used": model_id
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "model_used": model_id
            }
