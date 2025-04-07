"""
Image benchmarks for evaluating AI models
"""

IMAGE_BENCHMARKS = [
    {
        "id": "image_classification",
        "name": "Image Classification",
        "type": "image",
        "description": "This benchmark evaluates the model's ability to correctly identify objects, scenes, or concepts in images.",
        "metrics": ["accuracy", "precision", "recall", "f1_score"],
        "example": {
            "task_description": "Identify the main object or scene in this image and provide a classification label."
        }
    },
    {
        "id": "image_captioning",
        "name": "Image Captioning",
        "type": "image",
        "description": "This benchmark assesses the model's capability to generate accurate and descriptive captions for images.",
        "metrics": ["relevance", "accuracy", "bleu_score", "coherence"],
        "example": {
            "task_description": "Provide a detailed description of what you see in this image. The description should be 1-2 sentences long and capture the main elements and context."
        }
    },
    {
        "id": "visual_reasoning",
        "name": "Visual Reasoning",
        "type": "image",
        "description": "This benchmark tests the model's ability to understand relationships between objects and reason about visual information.",
        "metrics": ["accuracy", "reasoning_quality", "hallucination_rate"],
        "example": {
            "task_description": "Answer the following question based on the image: What spatial relationship exists between the objects in the image? How are they interacting or arranged?"
        }
    },
    {
        "id": "optical_character_recognition",
        "name": "Optical Character Recognition (OCR)",
        "type": "image",
        "description": "This benchmark evaluates the model's ability to extract and interpret text from images accurately.",
        "metrics": ["accuracy", "error_rate", "precision", "recall"],
        "example": {
            "task_description": "Extract all text visible in this image. Maintain the format and structure of the text as much as possible."
        }
    },
    {
        "id": "image_anomaly_detection",
        "name": "Image Anomaly Detection",
        "type": "image",
        "description": "This benchmark assesses the model's ability to identify unusual or anomalous elements in images.",
        "metrics": ["accuracy", "precision", "recall", "f1_score"],
        "example": {
            "task_description": "Identify any unusual, out-of-place, or anomalous elements in this image. Explain why they appear to be anomalous in the given context."
        }
    },
    {
        "id": "chart_interpretation",
        "name": "Chart and Graph Interpretation",
        "type": "image",
        "description": "This benchmark evaluates the model's ability to interpret and extract information from charts, graphs, and visual data representations.",
        "metrics": ["accuracy", "comprehensiveness", "inference_quality"],
        "example": {
            "task_description": "Interpret this chart/graph and provide a detailed summary of the information it presents. Include key trends, highest/lowest values, and any clear conclusions that can be drawn from the data."
        }
    },
    {
        "id": "object_counting",
        "name": "Object Counting and Quantification",
        "type": "image",
        "description": "This benchmark tests the model's ability to accurately count instances of specific objects or elements in images.",
        "metrics": ["accuracy", "error_rate"],
        "example": {
            "task_description": "Count the number of each type of object present in this image. Provide your answer as a list of object types and their counts."
        }
    }
]
