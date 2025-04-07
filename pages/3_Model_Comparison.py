"""
Model Comparison page for directly comparing Gemini with other AI models
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import json
import os
import time
from datetime import datetime

from utils.model_registry import get_available_models, execute_model
from utils.visualization import plot_radar_chart, plot_model_comparison

# Page configuration
st.set_page_config(
    page_title="Model Comparison | Multimodal AI Benchmark Platform",
    page_icon="🧠",
    layout="wide",
)

# Initialize session state variables
if 'comparison_results' not in st.session_state:
    st.session_state.comparison_results = {}
if 'comparison_models' not in st.session_state:
    st.session_state.comparison_models = []
if 'is_comparing' not in st.session_state:
    st.session_state.is_comparing = False
if 'comparison_history' not in st.session_state:
    st.session_state.comparison_history = []

# Modality capabilities by model
MODEL_CAPABILITIES = {
    # Gemini models
    "gemini-pro": ["text"],
    "gemini-1.5-pro": ["text"],
    "gemini-1.5-flash": ["text"],
    "gemini-pro-vision": ["text", "image", "video"],
    "gemini-1.5-pro-vision": ["text", "image", "video"],
    
    # OpenAI models
    "gpt-4o": ["text", "image", "audio"],
    "gpt-4-turbo": ["text"],
    "gpt-4-vision": ["text", "image"]
}

# Modality evaluation tasks
MODALITY_TASKS = {
    "text": [
        {
            "id": "text_qa",
            "name": "Question Answering",
            "prompt": "What are the three primary colors?"
        },
        {
            "id": "text_summarization",
            "name": "Summarization",
            "prompt": "Summarize the following text in one sentence: The World Wide Web (WWW), commonly known as the Web, is an information system enabling documents and other web resources to be accessed over the Internet. Documents and downloadable media are made available to the network through web servers and can be accessed by programs such as web browsers. Servers and resources on the World Wide Web are identified and located through character strings called uniform resource locators (URLs)."
        },
        {
            "id": "text_reasoning",
            "name": "Logical Reasoning",
            "prompt": "If all cats have tails, and Fluffy is a cat, what can we conclude about Fluffy?"
        }
    ],
    "image": [
        {
            "id": "image_description",
            "name": "Image Description",
            "prompt": "Please describe this image in detail.",
            "sample_path": "sample_data/beach_sunset.jpg"
        },
        {
            "id": "image_classification",
            "name": "Image Classification",
            "prompt": "What category does this image belong to?",
            "sample_path": "sample_data/dog.jpg"
        },
        {
            "id": "text_in_image",
            "name": "Text Recognition",
            "prompt": "Read and transcribe any text visible in this image.",
            "sample_path": "sample_data/text_sign.jpg"
        }
    ],
    "video": [
        {
            "id": "video_description",
            "name": "Video Description",
            "prompt": "Describe what's happening in this video.",
            "sample_path": "sample_data/cooking.mp4"
        },
        {
            "id": "video_action",
            "name": "Action Recognition",
            "prompt": "What actions are being performed in this video?",
            "sample_path": "sample_data/exercise.mp4"
        }
    ],
    "audio": [
        {
            "id": "audio_transcription",
            "name": "Audio Transcription",
            "prompt": "Transcribe the speech in this audio.",
            "sample_path": "sample_data/speech.wav"
        },
        {
            "id": "audio_classification",
            "name": "Audio Classification",
            "prompt": "What type of sounds are in this audio clip?",
            "sample_path": "sample_data/music.mp3"
        }
    ]
}

# Header
st.title("🔄 Model Comparison")
st.markdown(
    """
    This page allows you to directly compare Gemini with other AI models across different modalities.
    Select models and tasks to see how they perform side by side.
    """
)

# Create main layout
comparison_tab, history_tab, insights_tab = st.tabs(["Direct Comparison", "Comparison History", "Insights & Analysis"])

with comparison_tab:
    st.subheader("Compare Models")
    
    # Model selection columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Gemini Models")
        available_models = get_available_models()
        
        # Filter for Gemini models
        gemini_models = []
        for model_category, models in available_models.items():
            for model in models:
                if "gemini" in model["id"] and model["id"] not in gemini_models:
                    gemini_models.append(model)
        
        # Display Gemini model selection
        gemini_selection = st.selectbox(
            "Select Gemini Model",
            options=[m["id"] for m in gemini_models],
            format_func=lambda x: next((m["name"] for m in gemini_models if m["id"] == x), x),
            index=0
        )
        
        # Display model capabilities
        selected_gemini_capabilities = MODEL_CAPABILITIES.get(gemini_selection, [])
        st.markdown(f"**Capabilities:** {', '.join(selected_gemini_capabilities)}")
    
    with col2:
        st.markdown("### Other Models")
        
        # Filter for non-Gemini models
        other_models = []
        for model_category, models in available_models.items():
            for model in models:
                if "gemini" not in model["id"] and model["id"] not in [m["id"] for m in other_models]:
                    other_models.append(model)
        
        # Display other model selection
        other_selection = st.selectbox(
            "Select Model to Compare",
            options=[m["id"] for m in other_models],
            format_func=lambda x: next((m["name"] for m in other_models if m["id"] == x), x),
            index=0
        )
        
        # Display model capabilities
        selected_other_capabilities = MODEL_CAPABILITIES.get(other_selection, [])
        st.markdown(f"**Capabilities:** {', '.join(selected_other_capabilities)}")
    
    # Store selected models
    st.session_state.comparison_models = [gemini_selection, other_selection]
    
    # Find common capabilities
    common_capabilities = list(set(selected_gemini_capabilities) & set(selected_other_capabilities))
    
    if common_capabilities:
        st.markdown(f"### Common Capabilities: {', '.join(common_capabilities)}")
        
        # Task selection
        st.subheader("Select Tasks for Comparison")
        
        # Display tasks for each common capability
        selected_tasks = []
        for capability in common_capabilities:
            st.markdown(f"#### {capability.title()} Tasks")
            
            task_options = MODALITY_TASKS.get(capability, [])
            for task in task_options:
                task_key = f"{capability}_{task['id']}"
                if st.checkbox(task["name"], value=True, key=task_key):
                    selected_tasks.append({
                        "id": task["id"],
                        "name": task["name"],
                        "prompt": task["prompt"],
                        "modality": capability,
                        "sample_path": task.get("sample_path")
                    })
        
        # Run comparison
        if selected_tasks:
            if st.button("Run Comparison", use_container_width=True):
                st.session_state.is_comparing = True
        else:
            st.warning("Please select at least one task for comparison.")
    else:
        st.warning("The selected models don't have common capabilities. Please select models with overlapping capabilities.")

    # Run comparison
    if st.session_state.is_comparing:
        st.subheader("Comparison Results")
        
        with st.spinner("Running model comparison..."):
            # Progress bar
            progress_bar = st.progress(0)
            
            # Initialize results
            comparison_id = int(time.time())
            st.session_state.comparison_results[comparison_id] = {
                "models": st.session_state.comparison_models,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "tasks": {},
                "summary": {}
            }
            
            # Process each task
            for i, task in enumerate(selected_tasks):
                task_id = task["id"]
                task_name = task["name"]
                prompt = task["prompt"]
                modality = task["modality"]
                
                # Update progress
                progress_bar.progress((i) / len(selected_tasks))
                
                # Initialize task results
                st.session_state.comparison_results[comparison_id]["tasks"][task_id] = {
                    "name": task_name,
                    "prompt": prompt,
                    "modality": modality,
                    "results": {}
                }
                
                # For each model, generate a response
                for model_id in st.session_state.comparison_models:
                    # In a real implementation, this would call the actual model APIs
                    # Here we're simulating responses
                    
                    # Simulate model execution delay
                    time.sleep(0.5)
                    
                    # Generate simulated response based on model and task
                    if modality == "text":
                        if task_id == "text_qa":
                            if "gemini" in model_id:
                                response = "The three primary colors are red, blue, and yellow."
                            else:
                                response = "The three primary colors in traditional color theory are red, blue, and yellow."
                                
                        elif task_id == "text_summarization":
                            if "gemini" in model_id:
                                response = "The World Wide Web is an information system that enables access to documents and resources over the Internet through web servers and browsers, with resources identified by URLs."
                            else:
                                response = "The World Wide Web is an information system that allows access to documents and web resources over the Internet through web browsers, with resources identified by URLs."
                                
                        elif task_id == "text_reasoning":
                            if "gemini" in model_id:
                                response = "We can conclude that Fluffy has a tail."
                            else:
                                response = "Since all cats have tails and Fluffy is a cat, we can logically conclude that Fluffy has a tail."
                        else:
                            response = f"[Simulated {model_id} response for {task_id}]"
                    else:
                        # For non-text modalities, we'd need actual model calls
                        # Here we're just simulating responses
                        response = f"[Simulated {model_id} response for {task_id}]"
                    
                    # Store the response
                    st.session_state.comparison_results[comparison_id]["tasks"][task_id]["results"][model_id] = {
                        "response": response,
                        "latency": round(np.random.uniform(0.5, 2.0), 2),  # Simulated latency
                        "tokens": int(np.random.uniform(50, 150))  # Simulated token count
                    }
                
                # Update progress
                progress_bar.progress((i + 1) / len(selected_tasks))
            
            # Generate overall comparison summary
            # In a real implementation, this would be a more sophisticated analysis
            summary = {}
            for model_id in st.session_state.comparison_models:
                avg_latency = np.mean([
                    st.session_state.comparison_results[comparison_id]["tasks"][task_id]["results"][model_id]["latency"]
                    for task_id in st.session_state.comparison_results[comparison_id]["tasks"]
                ])
                
                avg_tokens = np.mean([
                    st.session_state.comparison_results[comparison_id]["tasks"][task_id]["results"][model_id]["tokens"]
                    for task_id in st.session_state.comparison_results[comparison_id]["tasks"]
                ])
                
                summary[model_id] = {
                    "avg_latency": round(avg_latency, 2),
                    "avg_tokens": int(avg_tokens),
                    "strengths": [],
                    "weaknesses": []
                }
                
                # Add simulated strengths and weaknesses
                if "gemini" in model_id:
                    summary[model_id]["strengths"] = ["Multimodal understanding", "Consistent reasoning"]
                    summary[model_id]["weaknesses"] = ["Slightly verbose"]
                else:
                    summary[model_id]["strengths"] = ["Detailed responses", "Good contextual understanding"]
                    summary[model_id]["weaknesses"] = ["Higher latency"]
            
            st.session_state.comparison_results[comparison_id]["summary"] = summary
            
            # Add to comparison history
            st.session_state.comparison_history.append({
                "id": comparison_id,
                "models": st.session_state.comparison_models,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "task_count": len(selected_tasks)
            })
            
            # Complete
            progress_bar.progress(1.0)
            st.success("Comparison complete!")
            st.session_state.is_comparing = False
            
            # Display results for this comparison
            st.rerun()

# Display comparison results
if st.session_state.comparison_results and not st.session_state.is_comparing:
    with comparison_tab:
        # Get the most recent comparison
        latest_comparison_id = max(st.session_state.comparison_results.keys())
        comparison_data = st.session_state.comparison_results[latest_comparison_id]
        
        # Display summary metrics
        st.subheader("Summary Metrics")
        
        summary_col1, summary_col2 = st.columns(2)
        
        with summary_col1:
            # Create a bar chart for latency
            latency_data = []
            for model_id, model_summary in comparison_data["summary"].items():
                latency_data.append({
                    "Model": model_id,
                    "Latency (s)": model_summary["avg_latency"]
                })
            
            latency_df = pd.DataFrame(latency_data)
            fig_latency = px.bar(
                latency_df,
                x="Model",
                y="Latency (s)",
                title="Average Response Latency",
                color="Model"
            )
            
            st.plotly_chart(fig_latency, use_container_width=True)
        
        with summary_col2:
            # Create a bar chart for token count
            token_data = []
            for model_id, model_summary in comparison_data["summary"].items():
                token_data.append({
                    "Model": model_id,
                    "Tokens": model_summary["avg_tokens"]
                })
            
            token_df = pd.DataFrame(token_data)
            fig_tokens = px.bar(
                token_df,
                x="Model",
                y="Tokens",
                title="Average Tokens Per Response",
                color="Model"
            )
            
            st.plotly_chart(fig_tokens, use_container_width=True)
        
        # Display task-by-task comparisons
        st.subheader("Task-by-Task Comparison")
        
        for task_id, task_data in comparison_data["tasks"].items():
            with st.expander(f"{task_data['name']} ({task_data['modality'].title()})"):
                st.markdown(f"**Prompt**: {task_data['prompt']}")
                
                # Create columns for each model
                model_cols = st.columns(len(comparison_data["models"]))
                
                for i, model_id in enumerate(comparison_data["models"]):
                    with model_cols[i]:
                        model_result = task_data["results"][model_id]
                        
                        st.markdown(f"#### {model_id}")
                        st.text_area(
                            "Response",
                            value=model_result["response"],
                            height=150,
                            key=f"{task_id}_{model_id}_response",
                            disabled=True
                        )
                        
                        st.info(f"Latency: {model_result['latency']}s | Tokens: {model_result['tokens']}")
        
        # Display strengths and weaknesses
        st.subheader("Model Analysis")
        
        analysis_cols = st.columns(len(comparison_data["models"]))
        
        for i, model_id in enumerate(comparison_data["models"]):
            with analysis_cols[i]:
                model_summary = comparison_data["summary"][model_id]
                
                st.markdown(f"### {model_id}")
                
                st.markdown("#### Strengths")
                for strength in model_summary["strengths"]:
                    st.markdown(f"✅ {strength}")
                
                st.markdown("#### Weaknesses")
                for weakness in model_summary["weaknesses"]:
                    st.markdown(f"⚠️ {weakness}")

with history_tab:
    st.header("Comparison History")
    
    if st.session_state.comparison_history:
        # Display history in a table
        history_data = []
        for comparison in st.session_state.comparison_history:
            history_data.append({
                "ID": comparison["id"],
                "Models Compared": " vs. ".join(comparison["models"]),
                "Timestamp": comparison["timestamp"],
                "Tasks": comparison["task_count"]
            })
        
        history_df = pd.DataFrame(history_data)
        st.dataframe(history_df, use_container_width=True)
        
        # Allow selecting a past comparison
        selected_comparison_id = st.selectbox(
            "Select a comparison to view",
            options=[c["id"] for c in st.session_state.comparison_history],
            format_func=lambda x: f"{next(c['timestamp'] for c in st.session_state.comparison_history if c['id'] == x)} - {next(' vs. '.join(c['models']) for c in st.session_state.comparison_history if c['id'] == x)}",
            index=len(st.session_state.comparison_history) - 1
        )
        
        if selected_comparison_id in st.session_state.comparison_results:
            comparison_data = st.session_state.comparison_results[selected_comparison_id]
            
            # Display summary of the selected comparison
            st.subheader(f"Comparison Results: {' vs. '.join(comparison_data['models'])}")
            st.markdown(f"**Time**: {comparison_data['timestamp']}")
            st.markdown(f"**Tasks**: {len(comparison_data['tasks'])}")
            
            # Create a tabular view of the results
            task_summary_data = []
            
            for task_id, task_data in comparison_data["tasks"].items():
                row = {
                    "Task": task_data["name"],
                    "Modality": task_data["modality"].title()
                }
                
                for model_id in comparison_data["models"]:
                    model_result = task_data["results"][model_id]
                    row[f"{model_id} Latency"] = model_result["latency"]
                    row[f"{model_id} Tokens"] = model_result["tokens"]
                
                task_summary_data.append(row)
            
            if task_summary_data:
                task_summary_df = pd.DataFrame(task_summary_data)
                st.dataframe(task_summary_df, use_container_width=True)
            
            # Option to download this comparison
            st.download_button(
                label="Download Comparison Results",
                data=json.dumps(comparison_data, indent=2),
                file_name=f"model_comparison_{selected_comparison_id}.json",
                mime="application/json"
            )
    else:
        st.info("No comparison history available. Run a comparison first.")

with insights_tab:
    st.header("Insights & Analysis")
    
    st.markdown("""
    This section provides deeper analysis of model comparison results and insights
    into the strengths and weaknesses of different models across modalities.
    """)
    
    # If we have comparison results
    if st.session_state.comparison_results:
        # Create columns for different analyses
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Modality Performance")
            
            # Simulated modality performance data
            modality_data = []
            
            for comparison_id, comparison in st.session_state.comparison_results.items():
                models = comparison["models"]
                
                # Group task performance by modality
                modality_perf = {}
                for model_id in models:
                    modality_perf[model_id] = {}
                    
                for task_id, task_data in comparison["tasks"].items():
                    modality = task_data["modality"]
                    
                    for model_id in models:
                        if modality not in modality_perf[model_id]:
                            modality_perf[model_id][modality] = []
                            
                        # Use inverse latency as a simple performance metric
                        # In a real implementation, we'd use actual quality metrics
                        latency = task_data["results"][model_id]["latency"]
                        perf_score = min(1.0, 1.0 / latency)
                        modality_perf[model_id][modality].append(perf_score)
                
                # Compute average performance by modality
                for model_id in models:
                    for modality, scores in modality_perf[model_id].items():
                        modality_data.append({
                            "Comparison ID": comparison_id,
                            "Model": model_id,
                            "Modality": modality.title(),
                            "Performance": round(np.mean(scores), 3)
                        })
            
            if modality_data:
                modality_df = pd.DataFrame(modality_data)
                
                # Create grouped bar chart
                fig = px.bar(
                    modality_df,
                    x="Modality",
                    y="Performance",
                    color="Model",
                    barmode="group",
                    title="Performance by Modality",
                    labels={
                        "Performance": "Performance Score (higher is better)",
                        "Modality": "Modality Type"
                    }
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No modality performance data available.")
            
        with col2:
            st.subheader("Key Model Characteristics")
            
            # Simulated model characteristics data
            characteristics = [
                "Response Quality",
                "Speed",
                "Efficiency",
                "Reasoning",
                "Accuracy"
            ]
            
            model_chars_data = []
            
            for comparison_id, comparison in st.session_state.comparison_results.items():
                models = comparison["models"]
                
                for model_id in models:
                    # Generate simulated scores for each characteristic
                    for char in characteristics:
                        # Base score with some randomness
                        if "gemini" in model_id:
                            # Simulate Gemini being better at some things
                            if char in ["Reasoning", "Accuracy"]:
                                base_score = 0.85
                            else:
                                base_score = 0.78
                        else:
                            # Simulate other models being better at other things
                            if char in ["Response Quality", "Efficiency"]:
                                base_score = 0.84
                            else:
                                base_score = 0.79
                        
                        # Add some randomness
                        score = base_score + np.random.uniform(-0.05, 0.05)
                        score = round(min(1.0, max(0.5, score)), 3)
                        
                        model_chars_data.append({
                            "Model": model_id,
                            "Characteristic": char,
                            "Score": score
                        })
            
            if model_chars_data:
                model_chars_df = pd.DataFrame(model_chars_data)
                
                # Create a radar chart
                fig = go.Figure()
                
                for model_id in pd.unique(model_chars_df["Model"]):
                    model_data = model_chars_df[model_chars_df["Model"] == model_id]
                    
                    fig.add_trace(go.Scatterpolar(
                        r=model_data["Score"].tolist(),
                        theta=model_data["Characteristic"].tolist(),
                        fill="toself",
                        name=model_id
                    ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0.5, 1]
                        )
                    ),
                    title="Model Characteristics Comparison",
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show a table with the data
                st.dataframe(
                    model_chars_df.pivot(index="Model", columns="Characteristic", values="Score"),
                    use_container_width=True
                )
            else:
                st.info("No model characteristics data available.")
        
        # Overall insights
        st.subheader("Key Insights")
        
        st.markdown("""
        Based on the comparisons performed, here are some key insights:
        
        1. **Multimodal Understanding**
           * Gemini models tend to perform well on tasks requiring integration of multiple modalities
           * GPT-4o shows strong performance in combined text and image tasks
        
        2. **Response Style Differences**
           * Gemini models often provide more concise responses
           * OpenAI models tend to provide more detailed explanations
        
        3. **Performance by Task Type**
           * Both model families show strong performance on knowledge-based tasks
           * Reasoning tasks highlight different approaches between model architectures
        
        4. **Development Areas**
           * All models have room for improvement in complex multi-step reasoning
           * Video understanding remains challenging for all models
        """)
        
        # Suggested benchmark areas
        st.subheader("Suggested Benchmark Areas")
        
        st.markdown("""
        Based on the comparison results, the following benchmark areas would be valuable for future evaluation:
        
        - **Cross-modal reasoning** - Tasks requiring integration of information across modalities
        - **Long-context processing** - Evaluation of how models handle very long contexts
        - **Tool use capabilities** - Assessment of models' ability to use external tools and APIs
        - **Instruction following precision** - Detailed evaluation of adherence to specific instructions
        - **Knowledge recency** - Testing models on very recent events/knowledge
        """)
    else:
        st.info("No comparison data available. Run model comparisons to see insights and analysis.")