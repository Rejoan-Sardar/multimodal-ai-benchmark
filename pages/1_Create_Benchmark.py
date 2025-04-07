"""
Create and manage custom benchmarks for multimodal AI evaluation
"""

import streamlit as st
import os
import pandas as pd
import json
import plotly.express as px
from datetime import datetime

from utils.benchmark_manager import (
    create_benchmark,
    load_benchmarks,
    update_benchmark,
    delete_benchmark,
    export_benchmark,
    import_benchmark
)

# Note: Page configuration is set in app.py

# Header
st.title("🔬 Create Custom Benchmarks")
st.markdown(
    """
    Design new benchmarks that focus on specific real-world tasks or scenarios, incorporating 
    multiple modalities (text, image, video, audio). Define tasks, evaluation metrics, 
    and prepare datasets for comprehensive multimodal AI evaluation.
    """
)

# Initialize session state variables with error handling
try:
    if 'custom_benchmarks' not in st.session_state:
        st.session_state.custom_benchmarks = load_benchmarks()
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = "create"
    if 'editing_benchmark' not in st.session_state:
        st.session_state.editing_benchmark = None
except Exception as e:
    st.error(f"Error initializing benchmark data: {str(e)}")
    # Provide fallback values
    if 'custom_benchmarks' not in st.session_state:
        st.session_state.custom_benchmarks = []
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = "create"
    if 'editing_benchmark' not in st.session_state:
        st.session_state.editing_benchmark = None

# Tab selection
tab1, tab2, tab3 = st.tabs(["Create New Benchmark", "Manage Benchmarks", "Import/Export"])

with tab1:
    st.header("Create a New Benchmark")
    
    # Form for creating a new benchmark
    with st.form("benchmark_creation_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            benchmark_name = st.text_input("Benchmark Name", placeholder="E.g., Multimodal Context Understanding")
            
            modality_type = st.selectbox(
                "Modality Type",
                ["text", "image", "video", "audio", "multimodal"],
                index=4,
                help="Select the primary modality or choose 'multimodal' for benchmarks that combine multiple modalities"
            )
            
            metrics = st.multiselect(
                "Evaluation Metrics",
                ["accuracy", "precision", "recall", "f1_score", "relevance", 
                 "coherence", "rouge_score", "bleu_score", "hallucination_rate", 
                 "inference_quality", "reasoning_quality", "error_rate",
                 "fluency", "engagement", "response_time", "custom_metric"],
                default=["accuracy", "relevance"],
                help="Select metrics that will be used to evaluate model performance on this benchmark"
            )
            
            if "custom_metric" in metrics:
                custom_metrics = st.text_input("Custom Metrics (comma-separated)", placeholder="E.g., context_retention, multi_hop_reasoning")
                if custom_metrics:
                    metrics.remove("custom_metric")
                    metrics.extend([m.strip() for m in custom_metrics.split(",")])
        
        with col2:
            benchmark_description = st.text_area(
                "Benchmark Description",
                placeholder="Describe what this benchmark evaluates and why it's important...",
                height=150
            )
            
            task_description = st.text_area(
                "Task Description",
                placeholder="Describe the specific task that models need to perform...",
                height=100
            )
            
            example_data = st.text_area(
                "Example Data (Optional)",
                placeholder="Provide example input data for this benchmark task...",
                height=100
            )
        
        st.markdown("### Modality-Specific Configuration")
        
        if modality_type == "text" or modality_type == "multimodal":
            st.markdown("**Text Configuration**")
            text_task_type = st.selectbox(
                "Text Task Type",
                ["Summarization", "Classification", "Question Answering", "Generation", "Translation", "Other"],
                index=0
            )
        
        if modality_type == "image" or modality_type == "multimodal":
            st.markdown("**Image Configuration**")
            image_task_type = st.selectbox(
                "Image Task Type",
                ["Classification", "Captioning", "Visual Reasoning", "OCR", "Object Detection", "Other"],
                index=0
            )
        
        if modality_type == "video" or modality_type == "multimodal":
            st.markdown("**Video Configuration**")
            video_task_type = st.selectbox(
                "Video Task Type",
                ["Action Recognition", "Temporal Understanding", "Caption Generation", "Anomaly Detection", "Other"],
                index=0
            )
        
        if modality_type == "audio" or modality_type == "multimodal":
            st.markdown("**Audio Configuration**")
            audio_task_type = st.selectbox(
                "Audio Task Type",
                ["Speech Recognition", "Speaker Identification", "Emotion Detection", "Audio Classification", "Other"],
                index=0
            )
        
        st.markdown("### Dataset Configuration")
        dataset_source = st.radio(
            "Dataset Source",
            ["Upload Custom Dataset", "Use Generated Examples", "Refer to External Dataset"],
            index=1,
            help="Specify how evaluation data will be provided for this benchmark"
        )
        
        if dataset_source == "Upload Custom Dataset":
            st.file_uploader("Upload Dataset Files", accept_multiple_files=True, type=["json", "csv", "txt", "png", "jpg", "mp3", "mp4", "wav"])
            st.info("Dataset upload functionality will be implemented in the next phase.")
        
        elif dataset_source == "Use Generated Examples":
            num_examples = st.number_input("Number of Examples to Generate", min_value=1, max_value=100, value=10)
            
        elif dataset_source == "Refer to External Dataset":
            dataset_url = st.text_input("Dataset URL or Reference", placeholder="E.g., https://huggingface.co/datasets/...")
        
        # Submit button
        submitted = st.form_submit_button("Create Benchmark")
        
        if submitted:
            if not benchmark_name or not benchmark_description or not metrics:
                st.error("Please fill in all required fields: Name, Description, and Metrics.")
            else:
                # Create the benchmark
                new_benchmark = create_benchmark(
                    name=benchmark_name,
                    description=benchmark_description,
                    modality_type=modality_type,
                    metrics=metrics,
                    example_data=example_data,
                    task_description=task_description
                )
                
                # Update session state
                st.session_state.custom_benchmarks.append(new_benchmark)
                
                st.success(f"Benchmark '{benchmark_name}' created successfully!")
                
                # Additional info about the benchmark
                st.json(new_benchmark)

with tab2:
    st.header("Manage Custom Benchmarks")
    
    # Display existing custom benchmarks
    if st.session_state.custom_benchmarks:
        # Create a DataFrame for displaying benchmarks
        benchmark_data = []
        for benchmark in st.session_state.custom_benchmarks:
            benchmark_data.append({
                "ID": benchmark["id"],
                "Name": benchmark["name"],
                "Type": benchmark["type"],
                "Metrics": ", ".join(benchmark["metrics"][:3]) + ("..." if len(benchmark["metrics"]) > 3 else ""),
                "Created": benchmark.get("created_at", "Unknown")
            })
        
        benchmark_df = pd.DataFrame(benchmark_data)
        st.dataframe(benchmark_df, use_container_width=True)
        
        # Benchmark selection for detailed view/editing
        selected_benchmark_id = st.selectbox(
            "Select a benchmark to view or edit",
            options=[b["id"] for b in st.session_state.custom_benchmarks],
            format_func=lambda x: next((b["name"] for b in st.session_state.custom_benchmarks if b["id"] == x), x)
        )
        
        # Find the selected benchmark
        selected_benchmark = next((b for b in st.session_state.custom_benchmarks if b["id"] == selected_benchmark_id), None)
        
        if selected_benchmark:
            st.subheader(f"Benchmark: {selected_benchmark['name']}")
            
            # View/Edit tabs
            view_edit_tab1, view_edit_tab2 = st.tabs(["View Details", "Edit Benchmark"])
            
            with view_edit_tab1:
                # Display benchmark details
                st.markdown(f"**Description:** {selected_benchmark['description']}")
                st.markdown(f"**Type:** {selected_benchmark['type']}")
                st.markdown(f"**Metrics:** {', '.join(selected_benchmark['metrics'])}")
                
                if "example" in selected_benchmark:
                    st.markdown("### Example")
                    if "task_description" in selected_benchmark["example"]:
                        st.markdown(f"**Task:** {selected_benchmark['example']['task_description']}")
                    if "input" in selected_benchmark["example"]:
                        st.text_area("Example Input", selected_benchmark["example"]["input"], height=100, disabled=True)
                
                # Actions
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Delete Benchmark", key="delete_btn", use_container_width=True):
                        if delete_benchmark(selected_benchmark_id):
                            st.session_state.custom_benchmarks = [b for b in st.session_state.custom_benchmarks if b["id"] != selected_benchmark_id]
                            st.success(f"Benchmark '{selected_benchmark['name']}' deleted.")
                            st.rerun()
                        else:
                            st.error("Failed to delete benchmark.")
                
                with col2:
                    export_format = st.selectbox("Export Format", ["JSON", "YAML"], disabled=True)
                    if st.button("Export Benchmark", key="export_btn", use_container_width=True):
                        export_path = export_benchmark(selected_benchmark_id)
                        if export_path:
                            with open(export_path, 'r') as f:
                                export_content = f.read()
                            
                            st.download_button(
                                label="Download Benchmark Definition",
                                data=export_content,
                                file_name=f"{selected_benchmark['name'].replace(' ', '_').lower()}_benchmark.json",
                                mime="application/json",
                                key="download_btn"
                            )
                        else:
                            st.error("Failed to export benchmark.")
            
            with view_edit_tab2:
                # Form for editing benchmark
                with st.form("edit_benchmark_form"):
                    edit_name = st.text_input("Benchmark Name", value=selected_benchmark["name"])
                    edit_description = st.text_area("Benchmark Description", value=selected_benchmark["description"], height=150)
                    
                    edit_metrics = st.multiselect(
                        "Evaluation Metrics",
                        ["accuracy", "precision", "recall", "f1_score", "relevance", 
                         "coherence", "rouge_score", "bleu_score", "hallucination_rate", 
                         "inference_quality", "reasoning_quality", "error_rate",
                         "fluency", "engagement", "response_time", "custom_metric"],
                        default=selected_benchmark["metrics"]
                    )
                    
                    if "example" in selected_benchmark and "task_description" in selected_benchmark["example"]:
                        edit_task = st.text_area(
                            "Task Description", 
                            value=selected_benchmark["example"]["task_description"], 
                            height=100
                        )
                    else:
                        edit_task = st.text_area("Task Description", height=100)
                    
                    if "example" in selected_benchmark and "input" in selected_benchmark["example"]:
                        edit_example = st.text_area(
                            "Example Input", 
                            value=selected_benchmark["example"]["input"], 
                            height=100
                        )
                    else:
                        edit_example = st.text_area("Example Input", height=100)
                    
                    # Submit edit button
                    edit_submitted = st.form_submit_button("Save Changes")
                    
                    if edit_submitted:
                        # Prepare updates
                        updates = {
                            "name": edit_name,
                            "description": edit_description,
                            "metrics": edit_metrics,
                            "last_modified": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                        
                        # Update example if provided
                        if edit_task or edit_example:
                            if "example" not in updates:
                                updates["example"] = {}
                            if edit_task:
                                updates["example"]["task_description"] = edit_task
                            if edit_example:
                                updates["example"]["input"] = edit_example
                        
                        # Update the benchmark
                        updated_benchmark = update_benchmark(selected_benchmark_id, updates)
                        if updated_benchmark:
                            # Update in session state
                            for i, b in enumerate(st.session_state.custom_benchmarks):
                                if b["id"] == selected_benchmark_id:
                                    st.session_state.custom_benchmarks[i] = updated_benchmark
                                    break
                            
                            st.success("Benchmark updated successfully!")
                        else:
                            st.error("Failed to update benchmark.")
    else:
        st.info("No custom benchmarks created yet. Use the 'Create New Benchmark' tab to get started.")

with tab3:
    st.header("Import/Export Benchmarks")
    
    # Import section
    st.subheader("Import Benchmark")
    import_format = st.selectbox("Import Format", ["JSON", "YAML"], disabled=True)
    uploaded_file = st.file_uploader("Upload Benchmark Definition", type=["json"])
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        temp_path = f"temp_import_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        if st.button("Import Benchmark"):
            imported_benchmark = import_benchmark(temp_path)
            if imported_benchmark:
                st.session_state.custom_benchmarks.append(imported_benchmark)
                st.success(f"Benchmark '{imported_benchmark['name']}' imported successfully!")
            else:
                st.error("Failed to import benchmark. Please check the file format.")
            
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    # Export all benchmarks
    st.subheader("Export All Benchmarks")
    if st.session_state.custom_benchmarks:
        if st.button("Export All Benchmarks"):
            # Create a combined export
            all_benchmarks_export = {
                "exported_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "benchmarks": st.session_state.custom_benchmarks
            }
            
            # Convert to JSON
            all_benchmarks_json = json.dumps(all_benchmarks_export, indent=2)
            
            # Provide download button
            st.download_button(
                label="Download All Benchmarks",
                data=all_benchmarks_json,
                file_name=f"all_custom_benchmarks_{int(datetime.now().timestamp())}.json",
                mime="application/json"
            )
    else:
        st.info("No custom benchmarks available to export.")

# Footer
st.markdown("---")
st.markdown(
    """
    **Custom Benchmark Creation Tool** | Part of the Multimodal AI Benchmark Platform
    """
)