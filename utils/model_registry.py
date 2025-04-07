import os
from utils.gemini_api import GeminiAPI
from utils.openai_api import OpenAIAPI

# Registry of model providers
MODEL_PROVIDERS = {}

def initialize_model_providers():
    """Initialize model provider clients"""
    global MODEL_PROVIDERS
    
    # Initialize Gemini if API key is available
    if os.getenv("GOOGLE_API_KEY"):
        try:
            gemini_api = GeminiAPI()
            MODEL_PROVIDERS["gemini"] = gemini_api
        except Exception as e:
            print(f"Error initializing Gemini API: {e}")
    
    # Initialize OpenAI if API key is available
    if os.getenv("OPENAI_API_KEY"):
        try:
            openai_api = OpenAIAPI()
            MODEL_PROVIDERS["openai"] = openai_api
        except Exception as e:
            print(f"Error initializing OpenAI API: {e}")
    
    # We're only using Gemini and OpenAI models

def get_available_models():
    """Get all available models from different providers"""
    if not MODEL_PROVIDERS:
        initialize_model_providers()
    
    all_models = {}
    
    # If no providers are available, provide dummy models for UI testing
    if not MODEL_PROVIDERS:
        return {
            "text": [
                {"id": "gemini-pro", "name": "Gemini Pro"},
                {"id": "gpt-4o", "name": "GPT-4o"}
            ],
            "multimodal": [
                {"id": "gemini-pro-vision", "name": "Gemini Pro Vision"},
                {"id": "gpt-4o", "name": "GPT-4o"}
            ]
        }
    
    # Collect models from all providers
    for provider_name, provider in MODEL_PROVIDERS.items():
        provider_models = provider.get_available_models()
        
        for model_type, models in provider_models.items():
            if model_type not in all_models:
                all_models[model_type] = []
            
            all_models[model_type].extend(models)
    
    return all_models

def get_model_provider(model_id):
    """Get the appropriate provider for a given model ID"""
    if not MODEL_PROVIDERS:
        initialize_model_providers()
    
    # Determine provider based on model ID prefix
    if model_id.startswith("gemini"):
        return MODEL_PROVIDERS.get("gemini")
    elif model_id.startswith("gpt"):
        return MODEL_PROVIDERS.get("openai")
    else:
        # Try to find any provider that has this model
        for provider in MODEL_PROVIDERS.values():
            all_provider_models = provider.get_available_models()
            for model_type, models in all_provider_models.items():
                if any(m["id"] == model_id for m in models):
                    return provider
    
    return None

def execute_model(model_id, task_type, **kwargs):
    """Execute a model with the appropriate provider"""
    provider = get_model_provider(model_id)
    
    if not provider:
        return {
            "error": f"No provider available for model {model_id}",
            "model_used": model_id
        }
    
    # Route to the appropriate method based on task type
    if task_type == "text":
        return provider.generate_text(model_id, **kwargs)
    elif task_type == "multimodal":
        return provider.generate_multimodal(model_id, **kwargs)
    elif task_type == "video":
        return provider.analyze_video(model_id, **kwargs)
    elif task_type == "audio":
        return provider.analyze_audio(model_id, **kwargs)
    else:
        return {
            "error": f"Unsupported task type: {task_type}",
            "model_used": model_id
        }
