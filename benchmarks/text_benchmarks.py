"""
Text benchmarks for evaluating AI models
"""

TEXT_BENCHMARKS = [
    {
        "id": "text_summarization",
        "name": "Text Summarization",
        "type": "text",
        "description": "This benchmark evaluates the model's ability to generate concise and accurate summaries of longer texts while preserving key information.",
        "metrics": ["accuracy", "relevance", "coherence", "rouge_score"],
        "example": {
            "input": "Scientists have discovered a new species of deep-sea fish that can survive the crushing pressures of the Mariana Trench, the deepest part of the world's oceans. The fish, dubbed the 'Mariana snailfish,' has several adaptations that allow it to live at depths of up to 8,000 meters (26,200 feet) below the surface. Its body contains a special protein that prevents the collapse of crucial cell components under extreme pressure. Additionally, it has developed a unique metabolism that functions with minimal oxygen. The discovery challenges previous assumptions about the limits of vertebrate life and opens new avenues for research into adaptations to extreme environments. Researchers from the Woods Hole Oceanographic Institution suggest that studying these adaptations could have applications in various fields, including medicine and materials science.",
            "expected_output": "Scientists discovered the 'Mariana snailfish' in the Mariana Trench that survives at depths of 8,000 meters using special proteins and unique metabolism. This finding challenges assumptions about vertebrate life limits and may impact medicine and materials science research."
        }
    },
    {
        "id": "sentiment_analysis",
        "name": "Sentiment Analysis",
        "type": "text",
        "description": "This benchmark tests the model's ability to correctly identify and categorize opinions expressed in text to determine the writer's attitude toward a particular topic or product.",
        "metrics": ["accuracy", "precision", "recall", "f1_score"],
        "example": {
            "input": "I was initially skeptical about the new restaurant, but the food was absolutely delicious and the service was impeccable. However, the prices were a bit steep for what you get, and the location is quite inconvenient with very limited parking. Overall, I'd probably visit again, but only for special occasions.",
            "expected_output": {
                "sentiment": "mixed",
                "positive_aspects": ["delicious food", "impeccable service"],
                "negative_aspects": ["steep prices", "inconvenient location", "limited parking"],
                "overall_sentiment_score": 0.6
            }
        }
    },
    {
        "id": "question_answering",
        "name": "Question Answering",
        "type": "text",
        "description": "This benchmark evaluates the model's ability to provide accurate answers to questions based on provided context.",
        "metrics": ["accuracy", "relevance", "hallucination_rate"],
        "example": {
            "input": "Context: The Great Barrier Reef, located in the Coral Sea off the coast of Queensland, Australia, is the world's largest coral reef system. It is composed of over 2,900 individual reefs and 900 islands stretching for over 2,300 kilometers. The reef is home to more than 1,500 species of fish, 400 species of coral, 4,000 species of mollusk, and 240 species of birds. The Great Barrier Reef was designated a UNESCO World Heritage Site in 1981.\n\nQuestion: How many species of fish live in the Great Barrier Reef?",
            "expected_output": "More than 1,500 species of fish live in the Great Barrier Reef."
        }
    },
    {
        "id": "text_classification",
        "name": "Text Classification",
        "type": "text",
        "description": "This benchmark tests the model's ability to categorize text into predefined classes or categories.",
        "metrics": ["accuracy", "precision", "recall", "f1_score"],
        "example": {
            "input": "The patient presents with fever of 101.3°F, persistent cough, and shortness of breath for the past 3 days. Chest X-ray shows patchy bilateral opacities. Blood oxygen saturation is 94% on room air. Patient reports recent travel to a region with known respiratory disease outbreak.",
            "expected_output": {
                "category": "Medical Report",
                "subcategory": "Respiratory Illness",
                "confidence": 0.95
            }
        }
    },
    {
        "id": "error_correction",
        "name": "Grammatical Error Correction",
        "type": "text",
        "description": "This benchmark evaluates the model's ability to identify and correct grammatical errors in text.",
        "metrics": ["accuracy", "precision", "recall", "f1_score"],
        "example": {
            "input": "The student have been studying for the test since last week, but she still don't feel prepared. She wish she would of started earlier. Her and her friend are planning to study together tonight, which might help them learns the material better.",
            "expected_output": {
                "corrected_text": "The student has been studying for the test since last week, but she still doesn't feel prepared. She wishes she would have started earlier. She and her friend are planning to study together tonight, which might help them learn the material better.",
                "corrections": [
                    {"original": "have been", "corrected": "has been", "error_type": "subject-verb agreement"},
                    {"original": "don't", "corrected": "doesn't", "error_type": "subject-verb agreement"},
                    {"original": "wish", "corrected": "wishes", "error_type": "subject-verb agreement"},
                    {"original": "would of", "corrected": "would have", "error_type": "idiom"},
                    {"original": "Her and her friend", "corrected": "She and her friend", "error_type": "pronoun case"},
                    {"original": "learns", "corrected": "learn", "error_type": "subject-verb agreement"}
                ]
            }
        }
    },
    {
        "id": "code_generation",
        "name": "Code Generation",
        "type": "text",
        "description": "This benchmark evaluates the model's ability to generate functional code based on natural language descriptions.",
        "metrics": ["accuracy", "functionality", "efficiency", "error_rate"],
        "example": {
            "input": "Write a Python function that takes a list of integers and returns the sum of all even numbers in the list.",
            "expected_output": "def sum_even_numbers(numbers):\n    return sum(num for num in numbers if num % 2 == 0)"
        }
    }
]
