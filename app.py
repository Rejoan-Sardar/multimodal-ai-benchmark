import streamlit as st
import os
import time
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

from utils.model_registry import get_available_models
from utils.file_processor import process_file
from utils.evaluator import evaluate_models
from utils.visualization import (
    plot_model_comparison, 
    plot_radar_chart, 
    plot_modality_performance
)

from benchmarks.text_benchmarks import TEXT_BENCHMARKS
from benchmarks.image_benchmarks import IMAGE_BENCHMARKS
from benchmarks.video_benchmarks import VIDEO_BENCHMARKS
from benchmarks.audio_benchmarks import AUDIO_BENCHMARKS
from benchmarks.knowledge_benchmarks import load_knowledge_benchmarks

# Page configuration
st.set_page_config(
    page_title="Multimodal AI Benchmark Platform",
    page_icon="🧠",
    layout="wide",
)

# Initialize session state variables
if 'current_benchmark' not in st.session_state:
    st.session_state.current_benchmark = None
if 'benchmark_results' not in st.session_state:
    st.session_state.benchmark_results = {}
if 'selected_models' not in st.session_state:
    st.session_state.selected_models = []
if 'benchmarks_run' not in st.session_state:
    st.session_state.benchmarks_run = []
if 'is_evaluating' not in st.session_state:
    st.session_state.is_evaluating = False
if 'show_api_config' not in st.session_state:
    st.session_state.show_api_config = False

# Header and description
st.title("🧠 Multimodal AI Benchmark Platform")
st.markdown(
    """
    This platform allows you to evaluate and compare Gemini with other multimodal AI models 
    across text, image, video, and audio tasks. Select benchmark tasks, choose models to evaluate,
    and visualize results for comprehensive performance analysis.
    """
)

# Add collapsible sections for Applications and Value
with st.expander("👉 Immediate Applications & Long-term Value"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Immediate Applications")
        st.markdown("""
        - Evaluate Gemini's current capabilities across modalities
        - Compare Gemini to other multimodal AI models (like GPT-4V)
        - Identify specific strengths and weaknesses in Gemini's performance
        """)
    
    with col2:
        st.subheader("Long-term Value")
        st.markdown("""
        - **Publication Opportunities:** Share findings through academic papers, conference presentations
        - **Benchmark Evolution:** Expand and refine the benchmark over time
        - **Industry Standard:** The benchmark could become a reference point for multimodal AI evaluation
        - **Development Guidance:** Results can inform future Gemini development priorities
        - **Research Contribution:** Advance the field's understanding of multimodal AI evaluation
        - **Commercial Decision-Making:** Help businesses assess if Gemini meets their specific requirements
        """)

# Sidebar for model selection and benchmark configuration
with st.sidebar:
    st.header("Configuration")
    
    # API Keys section
    st.subheader("API Keys")
    
    # Check for API keys
    google_api_key = os.environ.get("GOOGLE_API_KEY")
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    
    # Define API key status
    google_key_status = "✓" if google_api_key else "✗"
    openai_key_status = "✓" if openai_api_key else "✗"
    
    # Display API key status with colored status indicators
    col1, col2 = st.columns(2)
    with col1:
        if google_api_key:
            st.success(f"Google API Key: {google_key_status}")
        else:
            st.error(f"Google API Key: {google_key_status}")
            st.warning("⚠️ Gemini models unavailable")
        
    with col2:
        if openai_api_key:
            st.success(f"OpenAI API Key: {openai_key_status}")
        else:
            st.error(f"OpenAI API Key: {openai_key_status}")
            st.warning("⚠️ GPT models unavailable")
    
    # Button to trigger API key configuration
    if not (google_api_key and openai_api_key) or st.session_state.show_api_config:
        # Display more prominent warning 
        if not (google_api_key and openai_api_key):
            st.warning("⚠️ Missing API keys will limit model availability and benchmarking capabilities!")
            st.error("Without API keys, the platform will use SIMULATED DATA, not real API calls!")
        
        if st.button("Configure API Keys", type="primary", use_container_width=True) or st.session_state.show_api_config:
            # Reset the flag so we don't show this automatically next time
            st.session_state.show_api_config = False
            
            st.info("""
            ### API Key Information
            
            To use all models in this benchmark platform, you need to provide API keys for:
            
            1. **Google AI (for Gemini models)**
               - Required for: gemini-pro, gemini-pro-vision, gemini-1.5-pro, etc.
               - Get key at: https://ai.google.dev/
            
            2. **OpenAI (for GPT models)**
               - Required for: gpt-4o, gpt-4-turbo, gpt-4-vision, etc.
               - Get key at: https://platform.openai.com/account/api-keys
            
            These keys will be stored as environment variables for this session only and never shared.
            """)
            
            # Only show input fields for missing keys
            with st.form("api_key_form"):
                api_keys = {}
                
                if not google_api_key:
                    api_keys["GOOGLE_API_KEY"] = st.text_input(
                        "Google API Key (for Gemini models)", 
                        type="password",
                        help="Get yours at https://ai.google.dev/"
                    )
                
                if not openai_api_key:
                    api_keys["OPENAI_API_KEY"] = st.text_input(
                        "OpenAI API Key (for GPT models)", 
                        type="password",
                        help="Get yours at https://platform.openai.com/account/api-keys"
                    )
                
                submitted = st.form_submit_button("Save API Keys", type="primary", use_container_width=True)
                
                if submitted:
                    # Check if keys were provided
                    keys_updated = False
                    for key, value in api_keys.items():
                        if value:
                            os.environ[key] = value
                            keys_updated = True
                    
                    if keys_updated:
                        st.success("✅ API keys updated successfully!")
                        st.info("Reloading application to apply changes...")
                        time.sleep(1)  # Small delay for user to read the message
                        st.rerun()
                    else:
                        st.error("No API keys were provided. Please enter at least one API key.")
    
    st.divider()
    
    # Model selection
    st.subheader("Select Models")
    available_models = get_available_models()
    
    model_selection = {}
    for model_category, models in available_models.items():
        st.markdown(f"**{model_category} Models**")
        for model in models:
            key = f"{model_category}_{model['id']}"
            # Only set default selection if model hasn't been seen before
            # This prevents duplicates when a model appears in both text and multimodal categories
            if model['id'] not in [m.split('_')[1] for m, s in model_selection.items() if s]:
                checked = model['id'] == 'gemini-pro' or model['id'] == 'gpt-4o'  # Default selection
            else:
                checked = False
                
            model_selection[key] = st.checkbox(
                model['name'], 
                value=checked,
                key=key
            )
    
    # Update selected models
    st.session_state.selected_models = [
        model_id.split('_')[1] 
        for model_id, selected in model_selection.items() 
        if selected
    ]
    
    st.divider()
    
    # Benchmark selection
    st.subheader("Select Benchmark Type")
    benchmark_type = st.selectbox(
        "Modality", 
        ["Text", "Image", "Video", "Audio", "Multimodal"],
        index=0
    )
    
    # Load appropriate benchmarks based on type
    # Load knowledge benchmarks
    knowledge_benchmarks = load_knowledge_benchmarks()
    
    if benchmark_type == "Text":
        available_benchmarks = TEXT_BENCHMARKS + knowledge_benchmarks
    elif benchmark_type == "Image":
        available_benchmarks = IMAGE_BENCHMARKS
    elif benchmark_type == "Video":
        available_benchmarks = VIDEO_BENCHMARKS
    elif benchmark_type == "Audio":
        available_benchmarks = AUDIO_BENCHMARKS
    else:  # Multimodal
        available_benchmarks = (
            TEXT_BENCHMARKS + IMAGE_BENCHMARKS + 
            VIDEO_BENCHMARKS + AUDIO_BENCHMARKS +
            knowledge_benchmarks
        )
    
    # Display benchmark options
    if available_benchmarks:
        benchmark_options = [b["name"] for b in available_benchmarks]
        selected_benchmark_name = st.selectbox(
            "Select Benchmark", 
            benchmark_options
        )
        
        # Find the selected benchmark details
        st.session_state.current_benchmark = next(
            (b for b in available_benchmarks if b["name"] == selected_benchmark_name),
            None
        )
    else:
        st.warning("No benchmarks available for the selected type.")
        st.session_state.current_benchmark = None
    
    # Run benchmark button
    if st.session_state.selected_models and st.session_state.current_benchmark:
        # First check if we have the required API keys
        google_key_needed = any(model_id.startswith("gemini") for model_id in st.session_state.selected_models)
        openai_key_needed = any(model_id.startswith("gpt") or model_id.startswith("dall") for model_id in st.session_state.selected_models)
        
        missing_keys = []
        if google_key_needed and not google_api_key:
            missing_keys.append("Google API Key")
        if openai_key_needed and not openai_api_key:
            missing_keys.append("OpenAI API Key")
        
        # Show warnings if API keys are missing
        if missing_keys:
            st.error(f"⚠️ Cannot run benchmark without the required API keys: {', '.join(missing_keys)}")
            if st.button("Configure API Keys", key="config_before_run", type="primary", use_container_width=True):
                st.session_state.show_api_config = True
                st.rerun()
            st.markdown("Note: The benchmark will use real API calls and cannot run without proper authentication.")
        else:
            if st.button("Run Benchmark", use_container_width=True):
                st.session_state.is_evaluating = True
                # Track which benchmark is running
                benchmark_id = st.session_state.current_benchmark["id"]
                if benchmark_id not in st.session_state.benchmarks_run:
                    st.session_state.benchmarks_run.append(benchmark_id)
    
    st.divider()
    
    # Export results option
    if st.session_state.benchmark_results:
        st.subheader("Export Results")
        if st.button("Export as JSON", use_container_width=True):
            export_data = {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'results': st.session_state.benchmark_results
            }
            st.download_button(
                label="Download JSON",
                data=json.dumps(export_data, indent=2),
                file_name=f"benchmark_results_{int(time.time())}.json",
                mime="application/json",
                use_container_width=True
            )

# Main content area
if st.session_state.current_benchmark:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header(f"Benchmark: {st.session_state.current_benchmark['name']}")
        st.markdown(st.session_state.current_benchmark['description'])
        
        # Display benchmark details
        st.subheader("Benchmark Details")
        st.markdown(f"**Type:** {st.session_state.current_benchmark['type']}")
        st.markdown(f"**Metrics:** {', '.join(st.session_state.current_benchmark['metrics'])}")
        
        # Show example from the benchmark if available
        if 'example' in st.session_state.current_benchmark:
            st.subheader("Example")
            example = st.session_state.current_benchmark['example']
            
            if st.session_state.current_benchmark['type'] == 'text':
                st.text_area("Input Text", example['input'], height=100)
                
            elif st.session_state.current_benchmark['type'] == 'image':
                st.markdown("**Task Description:**")
                st.write(example['task_description'])
                st.markdown("*Image inputs will be uploaded during benchmarking*")
                
            elif st.session_state.current_benchmark['type'] == 'video':
                st.markdown("**Task Description:**")
                st.write(example['task_description'])
                st.markdown("*Video inputs will be uploaded during benchmarking*")
                
            elif st.session_state.current_benchmark['type'] == 'audio':
                st.markdown("**Task Description:**")
                st.write(example['task_description'])
                st.markdown("*Audio inputs will be uploaded during benchmarking*")
                
            elif st.session_state.current_benchmark['type'] == 'multimodal':
                st.markdown("**Task Description:**")
                st.write(example['task_description'])
                st.markdown("*Multiple inputs will be used during benchmarking*")
    
    with col2:
        st.subheader("Selected Models")
        if st.session_state.selected_models:
            for model_id in st.session_state.selected_models:
                st.markdown(f"- {model_id}")
        else:
            st.warning("No models selected. Please select at least one model from the sidebar.")

# Handle evaluation process
if st.session_state.is_evaluating and st.session_state.current_benchmark:
    benchmark_id = st.session_state.current_benchmark["id"]
    
    with st.spinner(f"Evaluating models on {st.session_state.current_benchmark['name']} benchmark..."):
        # Create evaluation container for real-time updates
        eval_container = st.container()
        progress_bar = st.progress(0)
        
        # Initialize status messages
        with eval_container:
            st.subheader("Benchmark Evaluation Progress")
            status_placeholder = st.empty()
            
        for i, model_id in enumerate(st.session_state.selected_models):
            # Update status
            with status_placeholder.container():
                st.info(f"Evaluating model: {model_id}")
                
            # Update progress
            progress_bar.progress((i + 0.1) / len(st.session_state.selected_models))
            
            # Start timer for overall evaluation
            eval_start_time = time.time()
            
            # Perform actual evaluation with real API calls
            try:
                results = evaluate_models(
                    model_id,
                    st.session_state.current_benchmark,
                )
            except Exception as e:
                # Display error and stop evaluation
                with status_placeholder.container():
                    st.error(f"❌ Error evaluating model {model_id}: {str(e)}")
                    if "API key" in str(e):
                        st.error("API key error detected. Please configure your API keys.")
                        if st.button("Configure API Keys", key=f"error_config_{int(time.time())}"):
                            st.session_state.show_api_config = True
                            st.rerun()
                    st.stop()  # Stop the evaluation process
            
            # Calculate evaluation time
            eval_time = time.time() - eval_start_time
            
            # Update progress
            progress_bar.progress((i + 0.8) / len(st.session_state.selected_models))
            
            # Store results with additional metadata
            if benchmark_id not in st.session_state.benchmark_results:
                st.session_state.benchmark_results[benchmark_id] = {}
            
            # Add metadata about the evaluation
            results["evaluation_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            results["evaluation_duration"] = round(eval_time, 2)
            results["model_id"] = model_id
            
            # Store in session state
            st.session_state.benchmark_results[benchmark_id][model_id] = results
            
            # Update status
            with status_placeholder.container():
                if "evaluation_type" in results and results["evaluation_type"] == "real":
                    st.success(f"✅ Model {model_id} evaluation complete (using real API calls)")
                    
                    # If we have API info, show it
                    if "_api_used" in results:
                        api_used = results["_api_used"].capitalize()
                        st.info(f"📡 API Used: {api_used} API")
                else:
                    st.warning(f"⚠️ Model {model_id} evaluation complete (using simulated results)")
                    
                    # Show reason for simulation if available
                    if "_simulation_reason" in results:
                        st.error(f"Simulation reason: {results['_simulation_reason']}")
                        
                        # If missing API key, show a button to configure
                        if "Missing" in results["_simulation_reason"] and "API key" in results["_simulation_reason"]:
                            if st.button("Configure API Keys", key=f"config_api_{model_id}_{int(time.time())}"):
                                st.session_state.show_api_config = True
                                st.rerun()
                
                # Show some key metrics (filtering out metadata fields)
                metric_results = {k: v for k, v in results.items() 
                                if isinstance(v, (int, float)) and 
                                k not in ["evaluation_duration"] and
                                not k.startswith("_")}
                
                metrics_str = ", ".join([f"{k}: {v:.3f}" for k, v in metric_results.items()])
                st.text(f"Metrics: {metrics_str}")
                st.text(f"Evaluation time: {results['evaluation_duration']}s")
                
            # Update final progress for this model
            progress_bar.progress((i + 1) / len(st.session_state.selected_models))
            
        # Finalize evaluation
        with eval_container:
            st.success("Benchmark evaluation complete for all selected models!")
            
        # Reset evaluation flag and refresh the UI
        st.session_state.is_evaluating = False
        time.sleep(2)  # Pause to show the success message
        st.rerun()

# Results display
if st.session_state.benchmark_results:
    st.header("Benchmark Results")
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["Performance Comparison", "Detailed Metrics", "Modality Analysis", "Applications & Value"])
    
    with tab1:
        st.subheader("Model Performance Comparison")
        
        # Filter benchmarks to show
        if st.session_state.benchmarks_run:
            benchmark_select = st.selectbox(
                "Select Benchmark to Visualize",
                options=st.session_state.benchmarks_run,
                format_func=lambda x: next((b['name'] for b in 
                                           TEXT_BENCHMARKS + IMAGE_BENCHMARKS + 
                                           VIDEO_BENCHMARKS + AUDIO_BENCHMARKS +
                                           load_knowledge_benchmarks()
                                           if b['id'] == x), x)
            )
            
            if benchmark_select in st.session_state.benchmark_results:
                results = st.session_state.benchmark_results[benchmark_select]
                
                # Create DataFrame for plotting
                df_list = []
                for model_id, model_results in results.items():
                    # Filter out metadata fields and only include metrics that are numeric values
                    for metric, value in model_results.items():
                        if (isinstance(value, (int, float)) and 
                            metric not in ["evaluation_duration", "evaluation_type", "model_id"] and
                            not metric.startswith("_")):
                            df_list.append({
                                "Model": model_id,
                                "Metric": metric,
                                "Score": value
                            })
                
                if df_list:
                    df = pd.DataFrame(df_list)
                    
                    # Bar chart comparing models across metrics
                    fig = plot_model_comparison(df)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Radar chart for overall performance
                    radar_fig = plot_radar_chart(df)
                    st.plotly_chart(radar_fig, use_container_width=True)
                else:
                    st.info("No results available for visualization.")
            else:
                st.info("No results available for the selected benchmark.")
        else:
            st.info("Run a benchmark to see results here.")
    
    with tab2:
        st.subheader("Detailed Metric Analysis")
        
        if st.session_state.benchmarks_run:
            metric_benchmark = st.selectbox(
                "Select Benchmark",
                options=st.session_state.benchmarks_run,
                format_func=lambda x: next((b['name'] for b in 
                                           TEXT_BENCHMARKS + IMAGE_BENCHMARKS + 
                                           VIDEO_BENCHMARKS + AUDIO_BENCHMARKS +
                                           load_knowledge_benchmarks()
                                           if b['id'] == x), x),
                key="metric_benchmark_select"
            )
            
            if metric_benchmark in st.session_state.benchmark_results:
                results = st.session_state.benchmark_results[metric_benchmark]
                
                # Create a nice table for the results
                table_data = []
                metrics = set()
                
                # Collect only valid metrics for visualization
                for model_id, model_results in results.items():
                    for metric, value in model_results.items():
                        if (isinstance(value, (int, float)) and 
                            metric not in ["evaluation_duration", "evaluation_type", "model_id"] and
                            not metric.startswith("_")):
                            metrics.add(metric)
                
                for model_id, model_results in results.items():
                    row = {"Model": model_id}
                    for metric in metrics:
                        row[metric] = model_results.get(metric, "N/A")
                    table_data.append(row)
                
                if table_data:
                    df_table = pd.DataFrame(table_data)
                    st.dataframe(df_table, use_container_width=True)
                    
                    # Allow focusing on a specific metric
                    if len(metrics) > 0:
                        selected_metric = st.selectbox(
                            "Focus on Metric",
                            options=list(metrics)
                        )
                        
                        # Create specific visualization for the selected metric
                        metric_data = []
                        for model_id, model_results in results.items():
                            if selected_metric in model_results:
                                metric_data.append({
                                    "Model": model_id,
                                    "Score": model_results[selected_metric]
                                })
                        
                        if metric_data:
                            metric_df = pd.DataFrame(metric_data)
                            fig = px.bar(
                                metric_df, 
                                x="Model", 
                                y="Score",
                                title=f"{selected_metric} Comparison",
                                color="Model"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info(f"No data available for {selected_metric}.")
                else:
                    st.info("No detailed metrics available.")
            else:
                st.info("No results available for the selected benchmark.")
        else:
            st.info("Run a benchmark to see detailed metrics here.")
    
    with tab3:
        st.subheader("Modality Performance Analysis")
        
        # Group benchmarks by modality type
        text_results = [bid for bid in st.session_state.benchmarks_run 
                      if bid in st.session_state.benchmark_results and 
                      next((b for b in TEXT_BENCHMARKS if b['id'] == bid), None)]
        
        image_results = [bid for bid in st.session_state.benchmarks_run 
                       if bid in st.session_state.benchmark_results and 
                       next((b for b in IMAGE_BENCHMARKS if b['id'] == bid), None)]
        
        video_results = [bid for bid in st.session_state.benchmarks_run 
                       if bid in st.session_state.benchmark_results and 
                       next((b for b in VIDEO_BENCHMARKS if b['id'] == bid), None)]
        
        audio_results = [bid for bid in st.session_state.benchmarks_run 
                       if bid in st.session_state.benchmark_results and 
                       next((b for b in AUDIO_BENCHMARKS if b['id'] == bid), None)]
        
        # Check if we have multiple modality types to compare
        modality_counts = {
            "Text": len(text_results),
            "Image": len(image_results),
            "Video": len(video_results),
            "Audio": len(audio_results)
        }
        
        if sum(modality_counts.values()) > 1:
            # Prepare data for modality comparison
            model_modality_data = []
            
            for model_id in st.session_state.selected_models:
                model_perf = {"Model": model_id}
                
                # Calculate average performance across benchmarks for each modality
                for modality, benchmarks in {
                    "Text": text_results,
                    "Image": image_results,
                    "Video": video_results,
                    "Audio": audio_results
                }.items():
                    if benchmarks:
                        scores = []
                        for bid in benchmarks:
                            if bid in st.session_state.benchmark_results and model_id in st.session_state.benchmark_results[bid]:
                                # Average numeric metrics (exclude metadata)
                                metric_values = [v for k, v in st.session_state.benchmark_results[bid][model_id].items() 
                                                if isinstance(v, (int, float)) and 
                                                k not in ["evaluation_duration", "evaluation_type", "model_id"] and
                                                not k.startswith("_")]
                                
                                if metric_values:
                                    avg_score = sum(metric_values) / len(metric_values)
                                    scores.append(avg_score)
                        
                        if scores:
                            model_perf[modality] = sum(scores) / len(scores)
                
                if len(model_perf) > 1:  # If we have at least one modality score
                    model_modality_data.append(model_perf)
            
            if model_modality_data:
                modality_df = pd.DataFrame(model_modality_data)
                
                # Create visualization
                fig = plot_modality_performance(modality_df)
                st.plotly_chart(fig, use_container_width=True)
                
                # Display the data table
                st.markdown("### Modality Performance Data")
                st.dataframe(modality_df, use_container_width=True)
            else:
                st.info("Not enough data to compare modality performance.")
        else:
            st.info("Run benchmarks across different modalities (text, image, video, audio) to see modality comparison.")
    
    with tab4:
        st.subheader("Benchmark Applications & Long-term Value")
        
        # Create two columns for Immediate Applications and Long-term Value
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Immediate Applications")
            st.markdown("""
            This benchmark platform provides several immediate applications:
            
            - **Evaluate Gemini's capabilities** across different modalities (text, image, video, audio)
            - **Compare Gemini to other multimodal AI models** like GPT-4V
            - **Identify specific strengths and weaknesses** in Gemini's performance
            
            By running comprehensive benchmarks across diverse tasks, you can gain
            insights into which model performs best for specific use cases and understand
            where each model excels or struggles.
            """)
            
            # Add visualization placeholder for applications
            st.image("https://storage.googleapis.com/images.geeksforgeeks.org/wp-content/uploads/20230804145809/AI-Model-Comparison.png", 
                     caption="Example of model comparison across various tasks")
        
        with col2:
            st.markdown("### Long-term Value")
            st.markdown("""
            The benchmark platform also offers substantial long-term value:
            
            - **Publication Opportunities:** Share benchmark findings through academic papers and conferences
            - **Benchmark Evolution:** Expand and refine the benchmark suite over time
            - **Industry Standard:** Establish a reference point for multimodal AI evaluation
            - **Development Guidance:** Inform future Gemini development priorities 
            - **Research Contribution:** Advance understanding of multimodal AI evaluation
            - **Commercial Decision-Making:** Help businesses assess model suitability for specific requirements
            
            This platform serves both research and practical business needs by providing
            objective, reproducible metrics across various modalities and tasks.
            """)
        
        # Include a unified section below for impact
        st.markdown("### Impact on AI Development")
        st.markdown("""
        Systematic benchmarking helps address key challenges in multimodal AI development:
        
        1. **Transparency:** Clear metrics revealing model capabilities and limitations
        2. **Objectivity:** Consistent evaluation allowing fair comparisons
        3. **Progress Tracking:** Measurable improvements across model versions
        4. **Gap Identification:** Highlighting areas needing further research
        
        As models like Gemini and GPT-4V evolve, this benchmark platform provides a
        structured way to track progress and guide development efforts.
        """)

# Footer
st.markdown("---")
st.markdown(
    """
    **Multimodal AI Benchmark Platform** | A tool for evaluating AI model performance across modalities.
    """
)
