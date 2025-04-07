"""
Audio benchmarks for evaluating AI models
"""

AUDIO_BENCHMARKS = [
    {
        "id": "speech_recognition",
        "name": "Speech Recognition",
        "type": "audio",
        "description": "This benchmark evaluates the model's ability to accurately transcribe spoken language from audio recordings into text.",
        "metrics": ["word_error_rate", "character_error_rate", "accuracy"],
        "example": {
            "task_description": "Transcribe all spoken content in this audio recording. Maintain proper punctuation and speaker separation if multiple speakers are present."
        }
    },
    {
        "id": "speaker_identification",
        "name": "Speaker Identification",
        "type": "audio",
        "description": "This benchmark tests the model's ability to identify and distinguish between different speakers in an audio recording.",
        "metrics": ["accuracy", "precision", "recall", "f1_score"],
        "example": {
            "task_description": "Identify how many unique speakers are in this audio recording. For each speaker, provide a brief description of their voice characteristics and the timestamps of when they are speaking."
        }
    },
    {
        "id": "sentiment_analysis_audio",
        "name": "Audio Sentiment Analysis",
        "type": "audio",
        "description": "This benchmark assesses the model's ability to determine the emotional tone and sentiment expressed in spoken content.",
        "metrics": ["accuracy", "precision", "recall", "f1_score"],
        "example": {
            "task_description": "Analyze the emotional tone of the speaker(s) in this audio. Identify the primary emotions expressed (e.g., happiness, anger, sadness, excitement) and provide a sentiment score on a scale from -1 (very negative) to +1 (very positive)."
        }
    },
    {
        "id": "audio_classification",
        "name": "Audio Classification",
        "type": "audio",
        "description": "This benchmark evaluates the model's ability to classify audio samples into predefined categories such as music genres, environmental sounds, or audio events.",
        "metrics": ["accuracy", "precision", "recall", "f1_score"],
        "example": {
            "task_description": "Classify this audio recording into the most appropriate category. Possible categories include: music (specify genre), speech, environmental sounds, animal sounds, mechanical/industrial sounds, etc. Provide confidence scores for your top 3 classifications."
        }
    },
    {
        "id": "audio_captioning",
        "name": "Audio Captioning",
        "type": "audio",
        "description": "This benchmark tests the model's ability to generate descriptive captions for audio content, capturing both speech and non-speech sounds.",
        "metrics": ["relevance", "accuracy", "comprehensiveness"],
        "example": {
            "task_description": "Create a detailed description of all sounds in this audio recording. Include both speech content and non-speech sounds (background noises, music, sound effects, etc.), and how they relate to each other temporally."
        }
    },
    {
        "id": "audio_question_answering",
        "name": "Audio Question Answering",
        "type": "audio",
        "description": "This benchmark evaluates the model's ability to answer questions about the content of audio recordings.",
        "metrics": ["accuracy", "relevance", "hallucination_rate"],
        "example": {
            "task_description": "Listen to this audio recording and answer the following questions: 1) What is the main topic being discussed? 2) How many speakers are there? 3) Is there any background noise or music? 4) What is the overall tone of the conversation or audio?"
        }
    }
]
