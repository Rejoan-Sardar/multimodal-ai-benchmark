"""
Video benchmarks for evaluating AI models
"""

VIDEO_BENCHMARKS = [
    {
        "id": "video_summarization",
        "name": "Video Summarization",
        "type": "video",
        "description": "This benchmark evaluates the model's ability to extract and summarize the key information and events from video content.",
        "metrics": ["relevance", "coherence", "comprehensiveness", "accuracy"],
        "example": {
            "task_description": "Based on the video frames provided, create a concise summary of the main events and content. Your summary should capture key activities, transitions, and important information in chronological order."
        }
    },
    {
        "id": "action_recognition",
        "name": "Action Recognition",
        "type": "video",
        "description": "This benchmark tests the model's ability to identify and describe human actions and activities occurring in video sequences.",
        "metrics": ["accuracy", "precision", "recall", "f1_score"],
        "example": {
            "task_description": "Identify all human actions and activities visible in this video. For each action, specify when it occurs (beginning, middle, end of the video or specific timestamp if available)."
        }
    },
    {
        "id": "temporal_event_detection",
        "name": "Temporal Event Detection",
        "type": "video",
        "description": "This benchmark assesses the model's ability to detect and locate specific events or transitions within a video timeline.",
        "metrics": ["accuracy", "precision", "recall", "temporal_precision"],
        "example": {
            "task_description": "Identify any notable events or transitions that occur in this video sequence. For each event, describe what happens and indicate when it occurs in the video timeline (beginning, middle, or end)."
        }
    },
    {
        "id": "video_question_answering",
        "name": "Video Question Answering",
        "type": "video",
        "description": "This benchmark evaluates the model's ability to answer specific questions about video content.",
        "metrics": ["accuracy", "relevance", "hallucination_rate"],
        "example": {
            "task_description": "Answer the following questions based on the video content: 1) What is the main activity shown in the video? 2) How many people are involved? 3) What is the setting or environment of the video?"
        }
    },
    {
        "id": "video_captioning",
        "name": "Video Captioning",
        "type": "video",
        "description": "This benchmark tests the model's ability to generate descriptive captions for video content that capture temporal dynamics and changes.",
        "metrics": ["relevance", "accuracy", "temporal_coherence", "comprehensiveness"],
        "example": {
            "task_description": "Generate a detailed caption for this video that describes not just what appears in the frames but also how actions and scenes progress over time. Your caption should capture the narrative flow of the video."
        }
    }
]
