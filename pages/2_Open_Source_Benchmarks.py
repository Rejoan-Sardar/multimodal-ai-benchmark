"""
Open Source Benchmarks page for evaluating models on established benchmarks
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import json
import os
import time
from datetime import datetime

from utils.model_registry import get_available_models, execute_model
from utils.evaluator import evaluate_models

# Open-source benchmarks
OPEN_SOURCE_BENCHMARKS = [
    {
        "id": "mmlu",
        "name": "MMLU (Massive Multitask Language Understanding)",
        "type": "text",
        "description": "A massive multitask benchmark designed to measure a model's ability to perform a wide variety of academic and professional tasks.",
        "metrics": ["accuracy", "precision", "recall"],
        "source": "https://arxiv.org/abs/2009.03300",
        "categories": ["World Knowledge", "STEM", "Humanities", "Social Sciences", "Others"]
    },
    {
        "id": "mmb",
        "name": "MMB (Multimodal Benchmark)",
        "type": "multimodal",
        "description": "A comprehensive benchmark for evaluating multimodal models across a diverse set of tasks including image understanding, reasoning, and generation.",
        "metrics": ["accuracy", "relevance", "hallucination_rate"],
        "source": "https://arxiv.org/abs/2307.12003",
        "categories": ["Vision-Language", "Audio-Language", "Video-Language"]
    },
    {
        "id": "hellaswag",
        "name": "HellaSwag",
        "type": "text",
        "description": "A commonsense reasoning benchmark for testing a model's ability to complete a sentence with the correct ending.",
        "metrics": ["accuracy", "relevance"],
        "source": "https://arxiv.org/abs/1905.07830",
        "categories": ["Commonsense Reasoning"]
    },
    {
        "id": "winogrande",
        "name": "WinoGrande",
        "type": "text",
        "description": "A large-scale dataset of Winograd Schema Challenge-style problems for commonsense reasoning.",
        "metrics": ["accuracy"],
        "source": "https://arxiv.org/abs/1907.10641",
        "categories": ["Commonsense Reasoning"]
    },
    {
        "id": "hateful_memes",
        "name": "Hateful Memes",
        "type": "multimodal",
        "description": "A benchmark for evaluating a model's ability to detect hate speech in multimodal content.",
        "metrics": ["accuracy", "precision", "recall", "f1_score"],
        "source": "https://arxiv.org/abs/2005.04790",
        "categories": ["Content Moderation"]
    },
    {
        "id": "vqa",
        "name": "Visual Question Answering (VQA)",
        "type": "multimodal",
        "description": "A benchmark for evaluating a model's ability to answer questions about images.",
        "metrics": ["accuracy", "relevance"],
        "source": "https://arxiv.org/abs/1505.00468",
        "categories": ["Vision-Language"]
    },
    {
        "id": "imagenet",
        "name": "ImageNet",
        "type": "image",
        "description": "A large-scale benchmark for image classification.",
        "metrics": ["accuracy", "precision", "recall", "f1_score"],
        "source": "https://www.image-net.org/",
        "categories": ["Computer Vision"]
    },
    {
        "id": "audioset",
        "name": "AudioSet",
        "type": "audio",
        "description": "A large-scale dataset of manually annotated audio events for audio classification.",
        "metrics": ["accuracy", "precision", "recall"],
        "source": "https://research.google.com/audioset/",
        "categories": ["Audio Classification"]
    }
]

# Page configuration
st.set_page_config(
    page_title="Open Source Benchmarks | Multimodal AI Benchmark Platform",
    page_icon="🧠",
    layout="wide",
)

# Initialize session state variables
if 'open_source_benchmark_results' not in st.session_state:
    st.session_state.open_source_benchmark_results = {}
if 'os_selected_models' not in st.session_state:
    st.session_state.os_selected_models = []
if 'os_benchmarks_run' not in st.session_state:
    st.session_state.os_benchmarks_run = []
if 'os_is_evaluating' not in st.session_state:
    st.session_state.os_is_evaluating = False

# Header
st.title("🔍 Open Source Benchmarks")
st.markdown(
    """
    This page allows you to evaluate models on established open-source benchmarks to compare 
    their performance in a standardized way.
    """
)

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    
    # Model selection
    st.subheader("Select Models")
    available_models = get_available_models()
    
    # Allow Gemini and OpenAI models to be selected
    model_selection = {}
    for model_category, models in available_models.items():
        if model_category in ["text", "multimodal"]:  # Only show applicable models
            st.markdown(f"**{model_category.title()} Models**")
            for model in models:
                key = f"os_{model_category}_{model['id']}"
                checked = model['id'] in ["gemini-1.5-pro", "gpt-4o"]  # Default selection
                model_selection[key] = st.checkbox(
                    model['name'], 
                    value=checked,
                    key=key
                )
    
    # Update selected models
    st.session_state.os_selected_models = [
        model_id.split('_')[2] 
        for model_id, selected in model_selection.items() 
        if selected
    ]
    
    st.divider()
    
    # Benchmark selection
    st.subheader("Select Benchmark Type")
    benchmark_type = st.selectbox(
        "Benchmark Modality", 
        ["All", "Text", "Image", "Audio", "Multimodal"],
        index=0,
        key="os_benchmark_type"
    )
    
    # Filter benchmarks based on selected type
    if benchmark_type == "All":
        filtered_benchmarks = OPEN_SOURCE_BENCHMARKS
    else:
        benchmark_type_lower = benchmark_type.lower()
        filtered_benchmarks = [b for b in OPEN_SOURCE_BENCHMARKS if b["type"] == benchmark_type_lower]
    
    # Display benchmark options
    if filtered_benchmarks:
        benchmark_options = [b["name"] for b in filtered_benchmarks]
        selected_benchmark_name = st.selectbox(
            "Select Benchmark", 
            benchmark_options,
            key="os_benchmark_select"
        )
        
        # Find the selected benchmark details
        selected_benchmark = next(
            (b for b in filtered_benchmarks if b["name"] == selected_benchmark_name),
            None
        )
    else:
        st.warning("No benchmarks available for the selected type.")
        selected_benchmark = None
    
    # Run benchmark button
    if st.session_state.os_selected_models and selected_benchmark:
        if st.button("Run Benchmark", use_container_width=True, key="os_run_btn"):
            st.session_state.os_is_evaluating = True
            # Track which benchmark is running
            benchmark_id = selected_benchmark["id"]
            if benchmark_id not in st.session_state.os_benchmarks_run:
                st.session_state.os_benchmarks_run.append(benchmark_id)
    
    st.divider()
    
    # Export results option
    if st.session_state.open_source_benchmark_results:
        st.subheader("Export Results")
        if st.button("Export as JSON", use_container_width=True, key="os_export_btn"):
            export_data = {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'open_source_benchmark_results': st.session_state.open_source_benchmark_results
            }
            st.download_button(
                label="Download JSON",
                data=json.dumps(export_data, indent=2),
                file_name=f"open_source_benchmark_results_{int(time.time())}.json",
                mime="application/json",
                use_container_width=True
            )

# Main content area
if selected_benchmark:
    st.header(f"Benchmark: {selected_benchmark['name']}")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown(selected_benchmark['description'])
        st.markdown(f"**Source:** [{selected_benchmark['source']}]({selected_benchmark['source']})")
        st.markdown(f"**Type:** {selected_benchmark['type'].title()}")
        st.markdown(f"**Metrics:** {', '.join(selected_benchmark['metrics'])}")
        
        # Categories
        if "categories" in selected_benchmark:
            st.markdown(f"**Categories:** {', '.join(selected_benchmark['categories'])}")
    
    with col2:
        st.subheader("Selected Models")
        if st.session_state.os_selected_models:
            for model_id in st.session_state.os_selected_models:
                st.markdown(f"- {model_id}")
        else:
            st.warning("No models selected. Please select at least one model from the sidebar.")
    
    # Information about the benchmark
    st.markdown("### About this Benchmark")
    
    # Different content based on benchmark ID
    if selected_benchmark["id"] == "mmlu":
        st.markdown("""
        **MMLU (Massive Multitask Language Understanding)** tests models across 57 subjects 
        spanning STEM, humanities, social sciences, and more. The benchmark requires both 
        broad factual knowledge and problem-solving abilities.
        
        The benchmark uses a multiple-choice format with questions from various domains including:
        - Mathematics (algebra, geometry, calculus)
        - Computer Science (algorithms, coding)
        - Physics, Chemistry, Biology
        - History, Philosophy, Law
        - Psychology, Sociology
        - And many more specialized fields
        
        A model's performance on MMLU is a strong indicator of its general knowledge and reasoning capabilities.
        """)
        
    elif selected_benchmark["id"] == "mmb":
        st.markdown("""
        **MMB (Multimodal Benchmark)** evaluates how well models can process and understand 
        information across different modalities (text, images, audio, video).
        
        The benchmark includes tasks such as:
        - Image captioning and visual question answering
        - Audio-video synchronization detection
        - Cross-modal retrieval
        - Multimodal reasoning
        
        This benchmark is particularly relevant for evaluating models like Gemini that are 
        designed with multimodal capabilities from the ground up.
        """)
        
    elif selected_benchmark["id"] == "vqa":
        st.markdown("""
        **Visual Question Answering (VQA)** tests a model's ability to answer natural language questions 
        about images. This requires both visual understanding and language comprehension.
        
        The benchmark includes:
        - Questions about objects and their attributes
        - Spatial relationships between objects
        - Counting and quantification
        - Questions requiring external knowledge
        
        VQA is a core test of vision-language integration capabilities.
        """)
    
    else:
        st.markdown(f"""
        This benchmark evaluates models on standardized tasks related to {selected_benchmark['type']} processing.
        For detailed information about methodology and scoring, visit the source link above.
        """)

# Handle evaluation process
if st.session_state.os_is_evaluating:
    benchmark_id = selected_benchmark["id"]
    
    with st.spinner(f"Evaluating models on {selected_benchmark['name']} benchmark..."):
        # Simulate progress
        progress_bar = st.progress(0)
        
        for i, model_id in enumerate(st.session_state.os_selected_models):
            time.sleep(1)  # Simulate work being done
            progress_bar.progress((i + 0.5) / len(st.session_state.os_selected_models))
            
            # Simulate evaluation (in a real implementation, this would call actual API evaluation)
            # We're using the same evaluate_models function but could be extended with specific
            # open-source benchmark functionality
            results = evaluate_models(
                model_id,
                selected_benchmark,
            )
            
            # Store results
            if benchmark_id not in st.session_state.open_source_benchmark_results:
                st.session_state.open_source_benchmark_results[benchmark_id] = {}
            
            st.session_state.open_source_benchmark_results[benchmark_id][model_id] = results
            
            progress_bar.progress((i + 1) / len(st.session_state.os_selected_models))
        
        st.success("Evaluation complete!")
        st.session_state.os_is_evaluating = False
        st.rerun()

# Results display
if st.session_state.open_source_benchmark_results:
    st.header("Benchmark Results")
    
    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["Performance Comparison", "Detailed Analysis", "Literature Comparison"])
    
    with tab1:
        st.subheader("Model Performance Comparison")
        
        # Filter benchmarks to show
        if st.session_state.os_benchmarks_run:
            benchmark_select = st.selectbox(
                "Select Benchmark to Visualize",
                options=st.session_state.os_benchmarks_run,
                format_func=lambda x: next((b['name'] for b in OPEN_SOURCE_BENCHMARKS if b['id'] == x), x),
                key="os_viz_benchmark_select"
            )
            
            if benchmark_select in st.session_state.open_source_benchmark_results:
                results = st.session_state.open_source_benchmark_results[benchmark_select]
                
                # Create DataFrame for plotting
                df_list = []
                for model_id, model_results in results.items():
                    for metric, value in model_results.items():
                        if metric != "error" and metric != "model_used":
                            df_list.append({
                                "Model": model_id,
                                "Metric": metric,
                                "Score": value
                            })
                
                if df_list:
                    df = pd.DataFrame(df_list)
                    
                    # Allow selecting specific metrics
                    metrics = df["Metric"].unique()
                    selected_metric = st.selectbox(
                        "Select Metric",
                        options=metrics,
                        key="os_selected_metric"
                    )
                    
                    # Filter for selected metric
                    df_metric = df[df["Metric"] == selected_metric]
                    
                    # Create bar chart
                    fig = px.bar(
                        df_metric,
                        x="Model",
                        y="Score",
                        title=f"{selected_metric.title()} Comparison",
                        color="Model",
                        labels={"Score": f"{selected_metric.title()} Score"},
                        text_auto='.3f'
                    )
                    
                    # Add a horizontal line for "State of the Art" performance (simulated)
                    benchmark_data = next((b for b in OPEN_SOURCE_BENCHMARKS if b['id'] == benchmark_select), None)
                    if benchmark_data:
                        # Simulate SOTA performance
                        sota_performance = 0.92  # This would be the actual SOTA value in a real implementation
                        fig.add_hline(
                            y=sota_performance,
                            line_dash="dash",
                            line_color="red",
                            annotation_text="State of the Art",
                            annotation_position="top right"
                        )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add heatmap for all metrics
                    st.subheader("All Metrics Heatmap")
                    
                    # Pivot the DataFrame for the heatmap
                    heatmap_df = df.pivot(index="Model", columns="Metric", values="Score")
                    
                    # Create heatmap
                    fig_heatmap = px.imshow(
                        heatmap_df,
                        text_auto='.3f',
                        title="Performance Across All Metrics",
                        color_continuous_scale="Viridis",
                        aspect="auto"
                    )
                    
                    st.plotly_chart(fig_heatmap, use_container_width=True)
                else:
                    st.info("No results available for visualization.")
            else:
                st.info("No results available for the selected benchmark.")
        else:
            st.info("Run a benchmark to see results here.")
    
    with tab2:
        st.subheader("Detailed Analysis")
        
        if st.session_state.os_benchmarks_run:
            benchmark_select = st.selectbox(
                "Select Benchmark",
                options=st.session_state.os_benchmarks_run,
                format_func=lambda x: next((b['name'] for b in OPEN_SOURCE_BENCHMARKS if b['id'] == x), x),
                key="os_detail_benchmark_select"
            )
            
            if benchmark_select in st.session_state.open_source_benchmark_results:
                # Get benchmark info
                benchmark_info = next((b for b in OPEN_SOURCE_BENCHMARKS if b['id'] == benchmark_select), None)
                if benchmark_info:
                    # Show benchmark categories for detailed analysis
                    if "categories" in benchmark_info:
                        st.markdown("#### Benchmark Categories")
                        
                        # In a real implementation, we would have category-specific results
                        # Here we'll simulate category results based on overall results
                        results = st.session_state.open_source_benchmark_results[benchmark_select]
                        
                        # Generate simulated category data
                        category_data = []
                        for category in benchmark_info["categories"]:
                            for model_id, model_results in results.items():
                                for metric in benchmark_info["metrics"]:
                                    if metric in model_results:
                                        # Add some variation to category scores
                                        base_score = model_results[metric]
                                        category_score = max(0, min(1, base_score + (hash(category) % 20 - 10) / 100))
                                        
                                        category_data.append({
                                            "Model": model_id,
                                            "Category": category,
                                            "Metric": metric,
                                            "Score": round(category_score, 3)
                                        })
                        
                        if category_data:
                            category_df = pd.DataFrame(category_data)
                            
                            # Select a specific metric for category analysis
                            unique_metrics = category_df["Metric"].unique()
                            selected_cat_metric = st.selectbox(
                                "Select Metric for Category Analysis",
                                options=unique_metrics,
                                key="os_cat_metric"
                            )
                            
                            # Filter for the selected metric
                            cat_metric_df = category_df[category_df["Metric"] == selected_cat_metric]
                            
                            # Create grouped bar chart
                            fig = px.bar(
                                cat_metric_df,
                                x="Category",
                                y="Score",
                                color="Model",
                                barmode="group",
                                title=f"{selected_cat_metric.title()} Performance by Category",
                                text_auto='.3f'
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Create radar chart for category performance
                            radar_data = []
                            for model in cat_metric_df["Model"].unique():
                                model_data = {"Model": model}
                                for category in cat_metric_df["Category"].unique():
                                    cat_score = cat_metric_df[(cat_metric_df["Model"] == model) & 
                                                         (cat_metric_df["Category"] == category)]["Score"].values
                                    if len(cat_score) > 0:
                                        model_data[category] = cat_score[0]
                                radar_data.append(model_data)
                            
                            radar_df = pd.DataFrame(radar_data)
                            
                            # Melt the dataframe for radar chart
                            radar_df_melted = pd.melt(
                                radar_df, 
                                id_vars=["Model"], 
                                var_name="Category", 
                                value_name="Score"
                            )
                            
                            # Create radar chart
                            fig_radar = px.line_polar(
                                radar_df_melted, 
                                r="Score", 
                                theta="Category", 
                                color="Model", 
                                line_close=True,
                                range_r=[0,1],
                                title=f"{selected_cat_metric.title()} Performance Radar Chart"
                            )
                            
                            st.plotly_chart(fig_radar, use_container_width=True)
                        else:
                            st.info("No category data available.")
                    else:
                        st.info("This benchmark does not have category-specific results.")
            else:
                st.info("No results available for the selected benchmark.")
        else:
            st.info("Run a benchmark to see detailed analysis here.")
    
    with tab3:
        st.subheader("Comparison with Published Results")
        
        st.markdown("""
        This section compares the performance of the evaluated models with published results from 
        research papers and official leaderboards.
        """)
        
        if st.session_state.os_benchmarks_run:
            benchmark_select = st.selectbox(
                "Select Benchmark",
                options=st.session_state.os_benchmarks_run,
                format_func=lambda x: next((b['name'] for b in OPEN_SOURCE_BENCHMARKS if b['id'] == x), x),
                key="os_lit_benchmark_select"
            )
            
            if benchmark_select in st.session_state.open_source_benchmark_results:
                # Get benchmark info
                benchmark_info = next((b for b in OPEN_SOURCE_BENCHMARKS if b['id'] == benchmark_select), None)
                
                if benchmark_info:
                    # Display published results (simulated)
                    st.markdown("#### Published Results")
                    
                    # Create a dataframe with simulated literature results
                    literature_models = [
                        "GPT-4 (2023)",
                        "PaLM 2 (2023)",
                        "Claude 2 (2023)",
                        "Llama 2 (2023)",
                        "SOTA (2024)"
                    ]
                    
                    # Primary metric for this benchmark
                    primary_metric = benchmark_info["metrics"][0]
                    
                    # Simulated scores for literature models
                    literature_scores = {
                        "mmlu": {
                            "GPT-4 (2023)": 0.86,
                            "PaLM 2 (2023)": 0.82,
                            "Claude 2 (2023)": 0.79,
                            "Llama 2 (2023)": 0.68,
                            "SOTA (2024)": 0.90
                        },
                        "hellaswag": {
                            "GPT-4 (2023)": 0.95,
                            "PaLM 2 (2023)": 0.91,
                            "Claude 2 (2023)": 0.92,
                            "Llama 2 (2023)": 0.85,
                            "SOTA (2024)": 0.96
                        },
                        "winogrande": {
                            "GPT-4 (2023)": 0.87,
                            "PaLM 2 (2023)": 0.83,
                            "Claude 2 (2023)": 0.81,
                            "Llama 2 (2023)": 0.77,
                            "SOTA (2024)": 0.89
                        },
                        "vqa": {
                            "GPT-4 (2023)": 0.80,
                            "PaLM 2 (2023)": 0.75,
                            "Claude 2 (2023)": 0.76,
                            "Llama 2 (2023)": 0.64,
                            "SOTA (2024)": 0.85
                        }
                    }
                    
                    # Default scores if benchmark isn't specifically defined
                    default_scores = {
                        "GPT-4 (2023)": 0.82,
                        "PaLM 2 (2023)": 0.78,
                        "Claude 2 (2023)": 0.77,
                        "Llama 2 (2023)": 0.70,
                        "SOTA (2024)": 0.88
                    }
                    
                    # Get the appropriate scores
                    scores = literature_scores.get(benchmark_select, default_scores)
                    
                    # Create dataframe with literature results
                    lit_df = pd.DataFrame({
                        "Model": literature_models,
                        "Score": [scores[model] for model in literature_models],
                        "Type": ["Published" for _ in literature_models]
                    })
                    
                    # Add current results
                    current_results = st.session_state.open_source_benchmark_results[benchmark_select]
                    for model_id, metrics in current_results.items():
                        if primary_metric in metrics:
                            lit_df = pd.concat([
                                lit_df,
                                pd.DataFrame({
                                    "Model": [model_id],
                                    "Score": [metrics[primary_metric]],
                                    "Type": ["Current Evaluation"]
                                })
                            ])
                    
                    # Create comparison chart
                    fig = px.bar(
                        lit_df,
                        x="Model",
                        y="Score",
                        color="Type",
                        title=f"Comparison with Published Results ({primary_metric.title()})",
                        text_auto='.3f'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add a table with the results
                    st.markdown("#### Detailed Comparison")
                    st.dataframe(lit_df, use_container_width=True)
                    
                    # Add interpretation
                    st.markdown("#### Interpretation")
                    st.markdown("""
                    The chart above compares your evaluation results with published results from research papers.
                    Note that there might be differences in testing methodology, prompting strategies, and model versions
                    that could account for performance variations.
                    
                    For the most accurate comparison, refer to the original publications linked in the benchmark description.
                    """)
            else:
                st.info("No results available for the selected benchmark.")
        else:
            st.info("Run a benchmark to compare with published results.")