"""
Knowledge benchmarks for evaluating AI models on factual recall, reasoning, and research capabilities
"""

import json
import os
import random

# Common Knowledge benchmark definition
COMMON_KNOWLEDGE_BENCHMARK = {
    "id": "common_knowledge",
    "name": "Common Knowledge Benchmark",
    "type": "text",
    "description": "Evaluates a model's ability to recall and reason about common factual knowledge across various domains.",
    "metrics": ["accuracy", "relevance", "hallucination_rate"],
    "tasks": [
        {
            "id": "history_knowledge",
            "input": "When was the Declaration of Independence signed?",
            "expected_output": "The Declaration of Independence was signed on July 4, 1776."
        },
        {
            "id": "science_knowledge",
            "input": "Explain the process of photosynthesis in simple terms.",
            "expected_output": "Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize foods with carbon dioxide and water. It converts light energy into chemical energy, releasing oxygen as a byproduct."
        },
        {
            "id": "geography_knowledge",
            "input": "What are the five largest countries in the world by land area?",
            "expected_output": "The five largest countries in the world by land area are Russia, Canada, United States, China, and Brazil."
        },
        {
            "id": "literature_knowledge",
            "input": "Who wrote 'Pride and Prejudice' and when was it published?",
            "expected_output": "Jane Austen wrote 'Pride and Prejudice,' which was published in 1813."
        },
        {
            "id": "pop_culture_knowledge",
            "input": "Who directed the movie 'Inception' and who was the main actor?",
            "expected_output": "Christopher Nolan directed 'Inception' and Leonardo DiCaprio played the main character, Dom Cobb."
        }
    ]
}

# Research Questions benchmark definition
RESEARCH_QUESTIONS_BENCHMARK = {
    "id": "research_questions",
    "name": "Research Questions Benchmark",
    "type": "text",
    "description": "Evaluates a model's ability to handle complex research questions requiring synthesis of information and critical thinking.",
    "metrics": ["relevance", "coherence", "hallucination_rate"],
    "tasks": [
        {
            "id": "climate_research",
            "input": "What are the most promising technologies for carbon capture and storage, and what are their limitations?",
            "expected_output": "The most promising carbon capture technologies include direct air capture (DAC), bioenergy with carbon capture and storage (BECCS), and enhanced weathering. Limitations include high energy requirements, cost, scalability challenges, and limited storage capacity. DAC currently costs $250-600 per ton of CO2, while BECCS faces land use competition and may not be carbon-negative over its lifecycle. Storage options like geological sequestration require suitable formations and monitoring for leakage."
        },
        {
            "id": "ai_ethics_research",
            "input": "What are the main ethical concerns surrounding the use of large language models, and how might these be addressed?",
            "expected_output": "Main ethical concerns include bias and discrimination, misinformation generation, lack of transparency, privacy issues, environmental impact, labor implications, and concentration of power. Addressing these requires diverse training data, bias detection tools, factual grounding techniques, explainable AI approaches, privacy-preserving methods, energy-efficient training, fair compensation for data contributors, and collaborative governance frameworks involving multiple stakeholders."
        },
        {
            "id": "education_research",
            "input": "How has remote learning during the COVID-19 pandemic affected educational outcomes, and what strategies have proven effective in mitigating learning loss?",
            "expected_output": "Remote learning during COVID-19 led to significant learning losses, particularly in mathematics and for disadvantaged students. The digital divide exacerbated inequalities. Effective mitigation strategies include high-dosage tutoring, extended learning time, targeted interventions for at-risk students, social-emotional support, interactive digital platforms, parent engagement programs, and hybrid learning models. Data shows high-intensity tutoring can help students gain 3-15 months of additional learning."
        },
        {
            "id": "economics_research",
            "input": "What are the potential economic impacts of widespread AI adoption, particularly on employment and inequality?",
            "expected_output": "Widespread AI adoption will likely cause job displacement in routine cognitive and manual tasks while creating new roles in AI development and oversight. This may increase productivity and GDP but worsen inequality through labor market polarization, with middle-skill jobs most vulnerable. Studies suggest 15-30% of jobs face high automation risk by 2030. Mitigating strategies include education reform emphasizing uniquely human skills, portable benefits systems, potential UBI programs, and progressive taxation of AI capital gains."
        },
        {
            "id": "health_research",
            "input": "What is the current state of research on longevity-extending interventions in humans, and what approaches show the most promise?",
            "expected_output": "Current promising longevity interventions include senolytics (drugs that clear senescent cells), NAD+ boosters, mTOR inhibitors like rapamycin, exercise mimetics, and partial cellular reprogramming. Human evidence is strongest for caloric restriction and intermittent fasting, which show metabolic benefits. Metformin and rapamycin have emerged from animal studies to early human trials, while senolytics (dasatinib/quercetin combination) show early promise. Epigenetic clocks are increasingly used as biomarkers. Challenges include translation from animal models, intervention timing, and measuring outcomes in long-lived species."
        }
    ]
}

def load_knowledge_benchmarks():
    """
    Load all knowledge benchmarks
    
    Returns:
        list: List of knowledge benchmark definitions
    """
    return [COMMON_KNOWLEDGE_BENCHMARK, RESEARCH_QUESTIONS_BENCHMARK]

def get_knowledge_benchmark(benchmark_id):
    """
    Get a specific knowledge benchmark by ID
    
    Args:
        benchmark_id: ID of the benchmark to retrieve
        
    Returns:
        dict: Benchmark definition or None if not found
    """
    all_benchmarks = load_knowledge_benchmarks()
    for benchmark in all_benchmarks:
        if benchmark["id"] == benchmark_id:
            return benchmark
    return None

def get_random_knowledge_question(benchmark_id=None, category=None):
    """
    Get a random knowledge question from a benchmark
    
    Args:
        benchmark_id: Optional ID of the benchmark to select from
        category: Optional category to filter by (e.g., "history", "science")
        
    Returns:
        dict: Question data with input prompt and expected output
    """
    if benchmark_id:
        benchmark = get_knowledge_benchmark(benchmark_id)
        if benchmark:
            tasks = benchmark["tasks"]
            if category:
                # Filter tasks by category if specified
                filtered_tasks = [t for t in tasks if category.lower() in t["id"].lower()]
                if filtered_tasks:
                    return random.choice(filtered_tasks)
            # No category filter or no matching tasks, return any task
            return random.choice(tasks)
    
    # No benchmark specified or not found, select from all benchmarks
    all_benchmarks = load_knowledge_benchmarks()
    benchmark = random.choice(all_benchmarks)
    return random.choice(benchmark["tasks"])