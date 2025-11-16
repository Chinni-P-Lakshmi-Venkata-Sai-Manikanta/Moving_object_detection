# Advanced AI-Powered Moving Object Detection System

A sophisticated, real-time moving object detection system that combines multiple AI models, advanced tracking algorithms, and comprehensive visualization capabilities.

## 🚀 Features

### Core Detection Capabilities
- **Multi-Model Ensemble Detection**: YOLO, DETR, and SAM integration for robust object detection
- **Advanced Motion Detection**: Optical flow, background subtraction, and frame differencing
- **Sophisticated Tracking**: Kalman filters, trajectory prediction, and multi-object tracking
- **Behavior Analysis**: Real-time behavior classification and anomaly detection

### Performance Optimization
- **GPU Acceleration**: CUDA support for high-performance processing
- **Multi-threading**: Background processing for real-time performance
- **Model Quantization**: Optimized models for faster inference
- **Memory Management**: Efficient memory usage and cleanup

### Visualization & Analytics
- **Real-time Visualization**: Bounding boxes, trajectories, and motion heatmaps
- **3D Trajectory Analysis**: Interactive 3D trajectory visualization
- **Performance Dashboard**: Real-time FPS, processing time, and system metrics
- **Data Export**: JSON, CSV, and video export capabilities

### Configuration & Monitoring
- **YAML Configuration**: Comprehensive configuration management
- **Real-time Parameter Tuning**: Hot-reloading of configuration parameters
- **Advanced Logging**: Structured logging with multiple output formats
- **Performance Monitoring**: Detailed performance metrics and analysis

## 📋 Requirements

### System Requirements
- Python 3.8+
- OpenCV 4.8+
- CUDA-capable GPU (recommended)
- 8GB+ RAM (16GB+ recommended)
- Webcam or video input

### Python Dependencies
```bash
pip install -r requirements.txt
```

## 🛠️ Installation

1. **Clone or download the project**
```bash
git clone <repository-url>
cd moving_object_detection
```

2. **Create virtual environment**
```bash
python -m venv motion_env
source motion_env/bin/activate  # On Windows: motion_env\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download AI models** (automatic on first run)
- YOLO models will be downloaded automatically
- DETR models will be downloaded automatically
- SAM models (optional) can be downloaded manually

## 🚀 Quick Start

### Live Detection
```bash
python advanced_main.py
```

### With Custom Camera
```bash
python advanced_main.py --camera 1
```

### Save Output Video
```bash
python advanced_main.py --save-video --output my_detection.mp4
```

### Analyze Video File
```bash
python advanced_main.py --analyze input_video.mp4 --analysis-output results/
```

## 📊 Configuration

The system uses a comprehensive YAML configuration file (`config.yaml`) that controls all aspects of the detection system:

### Detection Settings
```yaml
detection:
  confidence_threshold: 0.5
  nms_threshold: 0.4
  always_detect: false
  ensemble_weights:
    yolo: 0.6
    detr: 0.3
    sam: 0.1
```

### Motion Detection
```yaml
motion_detection:
  method: "optical_flow"  # optical_flow, background_subtraction, frame_differencing, hybrid
  sensitivity: 0.3
  min_area: 100
  temporal_consistency: true
```

### Tracking Configuration
```yaml
tracking:
  max_disappeared: 30
  max_distance: 50.0
  use_kalman: true
  use_deep_features: false
  trajectory_length: 50
```

## 🎮 Controls

### Live Detection Controls
- **Q**: Quit the application
- **S**: Save current detection state
- **H**: Show help information
- **C**: Clear trajectory data

### Configuration Hot-Reloading
The system supports real-time configuration updates. Modify `config.yaml` and the changes will be applied automatically.

## 📈 Performance Optimization

### GPU Acceleration
```yaml
performance:
  use_gpu: true
  batch_size: 1
  num_threads: 4
  memory_limit: 4096
```

### Model Optimization
```yaml
models:
  device: "auto"  # auto, cpu, cuda
  quantization: false
  model_optimization: true
```

## 📊 Visualization Features

### Real-time Visualization
- **Bounding Boxes**: Object detection with confidence scores
- **Trajectories**: Object movement paths with temporal fading
- **Predictions**: Kalman filter predictions (dashed lines)
- **Motion Heatmaps**: Motion intensity visualization
- **Performance Metrics**: FPS, processing time, object count

### Analysis Tools
- **3D Trajectory Plot**: Interactive 3D trajectory visualization
- **Trajectory Analysis**: Speed, direction, and distance analysis
- **Performance Dashboard**: Comprehensive performance monitoring

## 🔧 Advanced Usage

### Custom Model Integration
```python
from advanced_detector import AdvancedAIDetector

# Initialize with custom configuration
detector = AdvancedAIDetector("custom_config.yaml")

# Process single frame
result = detector.process_frame(frame)
```

### Behavior Analysis
```python
from tracker_utils import BehaviorAnalyzer

analyzer = BehaviorAnalyzer()
behaviors = analyzer.analyze_objects(tracks)
anomalies = analyzer.detect_anomalies(tracks)
```

### Custom Visualization
```python
from visualizer import AdvancedVisualizer

visualizer = AdvancedVisualizer()
vis_frame = visualizer.visualize_frame(frame, tracks, motion_regions)
```

## 📁 Project Structure

```
moving_object_detection/
├── advanced_main.py          # Main application
├── advanced_detector.py      # Core detection system
├── motion_detector.py        # Motion detection algorithms
├── tracker_utils.py         # Advanced tracking system
├── visualizer.py            # Visualization system
├── config_manager.py        # Configuration management
├── ai_object_detection.py   # AI description integration
├── config.yaml             # Configuration file
├── requirements.txt         # Python dependencies
├── README.md               # This file
└── logs/                   # Log files directory
```

## 🐛 Troubleshooting

### Common Issues

1. **Camera not detected**
   - Check camera permissions
   - Try different camera indices (0, 1, 2...)
   - Verify camera is not used by other applications

2. **Low performance**
   - Enable GPU acceleration in config
   - Reduce video resolution
   - Disable optional models (DETR, SAM)
   - Increase number of threads

3. **Model download issues**
   - Check internet connection
   - Verify model paths in config
   - Ensure sufficient disk space

4. **Memory issues**
   - Reduce batch size
   - Lower memory limit in config
   - Close other applications

### Performance Tips

1. **For Real-time Performance**:
   - Use YOLO only (disable DETR, SAM)
   - Set `always_detect: false`
   - Use `optical_flow` motion detection
   - Enable GPU acceleration

2. **For Maximum Accuracy**:
   - Enable all models (YOLO, DETR, SAM)
   - Use `hybrid` motion detection
   - Enable deep features for tracking
   - Use higher confidence thresholds

## 📊 Output Formats

### Video Output
- MP4 format with H.264 encoding
- Configurable resolution and frame rate
- Optional audio track preservation

### Data Export
- **JSON**: Detection results, trajectories, performance metrics
- **CSV**: Trajectory data for analysis
- **Images**: Trajectory analysis plots, performance dashboards

### Logging
- **File Logging**: Structured logs with timestamps
- **Console Logging**: Real-time status updates
- **Performance Logging**: FPS, processing time, memory usage

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Ultralytics**: YOLO models and framework
- **Hugging Face**: DETR models and transformers
- **OpenCV**: Computer vision algorithms
- **PyTorch**: Deep learning framework
- **Supervision**: Object tracking utilities

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the configuration options
3. Check the logs for error messages
4. Create an issue with detailed information
5. For support call 7989132992

---

**Note**: This system requires significant computational resources. For optimal performance, use a CUDA-capable GPU with at least 8GB VRAM and 16GB system RAM.
