# Multimodal AI Benchmark Platform

A comprehensive platform for evaluating and comparing Google's Gemini with other multimodal AI models (like GPT-4V) across text, image, video, and audio tasks. The platform enables standardized testing and visualization of model performance across various benchmarking scenarios.

![Multimodal AI Benchmark Platform](generated-icon.avif)

## Features

- **Multi-Model Evaluation**: Test and compare multiple AI models including:
  - Google's Gemini family (Pro, Flash, Vision)
  - OpenAI's GPT-4 family (GPT-4o, GPT-4 Turbo, Vision)
  - Text, image, video, and audio generation models

- **Comprehensive Modalities**:
  - Text: Knowledge Q&A, code generation, creative writing
  - Image: Object recognition, visual reasoning, chart analysis
  - Video: Scene understanding, action recognition
  - Audio: Speech understanding, emotion recognition
  - Multimodal: Combined tasks using multiple input types

- **Advanced Analytics**:
  - Standardized benchmark protocols and metrics
  - Performance visualization with interactive charts
  - Detailed metric breakdowns and comparisons
  - Export results for further analysis

- **Custom Benchmarking**:
  - Create and manage custom benchmarks
  - Import/export benchmark definitions
  - Run models against established open source benchmarks

## Getting Started

### Prerequisites

- Python 3.9 or higher
- API keys:
  - Google API key for Gemini models (get from [Google AI Studio](https://ai.google.dev/))
  - OpenAI API key for GPT models (get from [OpenAI Platform](https://platform.openai.com/api-keys))
- Libraries for multimodal processing:
  - Image: OpenCV, PIL
  - Video: ffmpeg
  - Audio: librosa, SpeechRecognition

### Installation

1. Clone this repository
   ```bash
   git clone https://github.com/yourusername/multimodal-ai-benchmark.git
   cd multimodal-ai-benchmark
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   
3. Set up API keys as environment variables:
   ```bash
   export GOOGLE_API_KEY='your_gemini_api_key'
   export OPENAI_API_KEY='your_openai_api_key'
   ```
   Or configure them within the application interface.

### Running the Application

Launch the Streamlit application:
```bash
streamlit run app.py --server.port 5000 --server.fileWatcherType none --server.maxUploadSize 10 --server.maxMessageSize 50
```

The application will be available at `http://localhost:5000`

## Benchmark Types

The platform includes several pre-configured benchmark types:

### Text Benchmarks
- General Knowledge Q&A
- Code Completion
- Creative Writing
- Research Capabilities
- Logical Reasoning

### Image Benchmarks
- Object Recognition
- Visual Reasoning
- Chart and Graph Analysis

### Video Benchmarks
- Scene Understanding
- Action Recognition

### Audio Benchmarks
- Speech Understanding
- Emotion Recognition

### Multimodal Benchmarks
- Image Captioning
- Visual Question Answering

## Usage Guide

1. **Configure API Keys**:
   - Enter your Google and OpenAI API keys in the sidebar
   
2. **Select Models**:
   - Choose one or more models to evaluate from the sidebar
   
3. **Choose Benchmark Type**:
   - Select modality (Text, Image, Video, Audio, Multimodal)
   - Pick a specific benchmark
   
4. **Run Benchmark**:
   - Click "Run Benchmark" to start evaluation
   - For image/video/audio benchmarks, you'll be prompted to upload files
   
5. **View Results**:
   - See performance comparisons across models
   - Explore detailed metrics
   - Export results as needed

## Development

### Project Structure

```
multimodal-ai-benchmark/
├── app.py                     # Main Streamlit application
├── utils/                     # Utility modules
│   ├── model_registry.py      # Model availability and execution
│   ├── gemini_api.py          # Google Gemini API integration
│   ├── openai_api.py          # OpenAI API integration
│   ├── evaluator.py           # Benchmark evaluation engine
│   ├── visualization.py       # Data visualization tools
│   ├── file_processor.py      # Process uploaded files (images, videos, etc.)
│   └── benchmark_manager.py   # Custom benchmark creation & management
├── benchmarks/                # Benchmark definitions
│   ├── text_benchmarks.py     # Text benchmark tasks
│   ├── image_benchmarks.py    # Image benchmark tasks
│   ├── video_benchmarks.py    # Video benchmark tasks
│   ├── audio_benchmarks.py    # Audio benchmark tasks
│   └── knowledge_benchmarks.py # Knowledge benchmark tasks
├── pages/                     # Streamlit multi-page components
│   ├── 1_Create_Benchmark.py  # Custom benchmark creation
│   ├── 2_Open_Source_Benchmarks.py # Established benchmarks
│   └── 3_Model_Comparison.py  # Direct model comparison
├── sample_data/               # Example benchmark data
│   └── multimodal_benchmark.json # Sample benchmark definitions
└── docs/                      # Documentation
    └── benchmark_documentation.md # Detailed documentation
```

### Extending the Platform

To add new benchmark types:
1. Create a new file in the `benchmarks/` directory
2. Define benchmark tasks and metrics
3. Import and register in `app.py`

To add support for new models:
1. Create an API client in the `utils/` directory
2. Implement standard interface methods
3. Register in `model_registry.py`

## Documentation

Comprehensive documentation is available in the `docs/` directory. See:
- [Benchmark Documentation](docs/benchmark_documentation.md) for detailed usage instructions


## Acknowledgements

- Google Gemini API
- OpenAI API
- Streamlit framework
- The open source AI evaluation community
