# Car Brand and Model Detector from Vehicle Parts

This project presents a comprehensive comparative study of vehicle make and model recognition using three distinct approaches: YOLO-based full vehicle detection, EfficientNet-based headlight classification, and Vision Transformer (ViT)-based headlight classification.

## 🚗 Project Overview

While traditional vehicle recognition systems rely on full vehicle imagery, this work explores the novel approach of using only vehicle headlight regions for fine-grained classification, motivated by the distinctive design patterns that manufacturers embed in headlight assemblies.

### Key Features

- **Three Model Approaches**: YOLO, EfficientNet, and ViT implementations
- **Novel Headlight-Based Recognition**: Component-based vehicle classification
- **Stanford Cars Dataset**: 196 car make-model combinations
- **Comprehensive Evaluation**: Detailed performance comparisons

## 📊 Performance Results

| Model | Approach | Top-1 Accuracy | Training Time |
|-------|----------|----------------|---------------|
| YOLOv8n | Full Vehicle | 85% | ~4 hours |
| EfficientNet-B0 | Headlight-based | 48% | ~30 hours |
| ViT-Base/16 | Headlight-based | 85% | ~8 hours |

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- 8GB+ GPU memory

### Dependencies

Create a virtual environment and install dependencies:

```bash
# Create virtual environment
python -m venv car_detector_env

# Activate virtual environment
# Windows:
car_detector_env\Scripts\activate
# Linux/Mac:
source car_detector_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements.txt

```txt
# Deep Learning Frameworks
torch>=1.13.0
torchvision>=0.14.0
timm>=0.6.12
ultralytics>=8.0.0

# Computer Vision
opencv-python>=4.7.0
pillow>=9.4.0

# Data Science
numpy>=1.21.0
pandas>=1.5.0
scikit-learn>=1.2.0
matplotlib>=3.6.0
seaborn>=0.11.0

# Progress and Logging
tqdm>=4.64.0
logging

# Model Utilities
pyyaml>=6.0
pathlib

# Additional ML Tools
collections
argparse
json
time
random
```

## 📁 Project Structure

```
Car-Brand-and-Model-Detector-from-Vehicle-Parts/
├── README.md
├── requirements.txt
├── run.py                              # Main YOLO script (legacy)
├── stanford_cars_detector.py           # Optimized YOLO implementation
├── headlight_classifier.py            # EfficientNet headlight classifier
├── vit_headlight_classifier.py        # ViT headlight classifier
├── dataset_utils.py                   # Dataset handling utilities
├── yolo_fine_tuner.py                 # YOLO fine-tuning utilities
├── headlight_dataset_creator.py       # Headlight extraction tools
├── enhanced_headlight_dataset/         # Headlight dataset
│   ├── train/
│   │   ├── train_headlight_mapping.csv
│   │   └── headlight_images/
│   └── test/
│       ├── test_headlight_mapping.csv
│       └── headlight_images/
├── stanford_cars_yolo/                # YOLO dataset
│   └── data/
│       ├── dataset.yaml
│       ├── classes.txt
│       ├── train/
│       ├── val/
│       └── test/
├── headlight_classifier_output/       # EfficientNet outputs
├── vit_headlight_classifier_output/   # ViT outputs
└── runs/                              # YOLO training outputs
```

## 🚀 Quick Start

### 1. YOLO-Based Full Vehicle Detection

```bash
# Train YOLO model on Stanford Cars
python stanford_cars_detector.py --train --epochs 100 --batch-size 16

# Predict on single image
python stanford_cars_detector.py --predict path/to/car_image.jpg --conf 0.25

# Process video
python stanford_cars_detector.py --video path/to/video.mp4 --save

# Real-time camera demo
python stanford_cars_detector.py --camera

# Batch processing
python stanford_cars_detector.py --batch path/to/image_folder --save
```

### 2. EfficientNet Headlight Classification

```bash
# Train EfficientNet model
python headlight_classifier.py --train --output headlight_classifier_output

# Single image prediction
python headlight_classifier.py --predict path/to/headlight.jpg

# Batch prediction
python headlight_classifier.py --batch-predict path/to/headlight_folder

# Evaluate model
python headlight_classifier.py --evaluate
```

### 3. ViT Headlight Classification

```bash
# Train ViT model (different sizes available)
python vit_headlight_classifier.py --train --model-size base

# Available model sizes: tiny, small, base
python vit_headlight_classifier.py --train --model-size tiny  # Faster training

# Prediction
python vit_headlight_classifier.py --predict path/to/headlight.jpg

# Evaluate and compare with EfficientNet
python vit_headlight_classifier.py --evaluate
python vit_headlight_classifier.py --compare
```

## 📋 Detailed Usage

### Stanford Cars YOLO Detector

The `stanford_cars_detector.py` provides a complete implementation for vehicle detection and classification:

**Training Parameters:**
- `--epochs`: Number of training epochs (default: 100)
- `--batch-size`: Training batch size (default: 16)
- `--img-size`: Input image size (default: 640)
- `--model-size`: YOLO variant [n,s,m,l,x] (default: n)

**Inference Parameters:**
- `--conf`: Confidence threshold (default: 0.25)
- `--save`: Save annotated results
- `--no-show`: Disable result visualization

### EfficientNet Headlight Classifier

The `headlight_classifier.py` implements CNN-based headlight recognition:

**Key Features:**
- EfficientNet-B0 backbone with transfer learning
- Custom classification head with batch normalization
- Balanced sampling for imbalanced classes
- Advanced data augmentation strategies
- Early stopping and learning rate scheduling

**Training Configuration:**
```python
config = {
    'model_name': 'efficientnet_b0',
    'batch_size': 32,
    'num_epochs': 50,
    'learning_rate': 3e-4,
    'dropout_rate': 0.2,
    'image_size': 224,
    'hidden_size': 512
}
```

### ViT Headlight Classifier

The `vit_headlight_classifier.py` implements transformer-based recognition:

**Key Features:**
- Vision Transformer (ViT-Base/16) with patch-based processing
- Attention mechanisms for fine-grained pattern recognition
- Mixup and CutMix augmentation strategies
- Gradient checkpointing for memory optimization
- Advanced regularization techniques

**Training Configuration:**
```python
config = {
    'model_name': 'vit_base_patch16_224',
    'batch_size': 16,
    'num_epochs': 60,
    'learning_rate': 1e-4,
    'weight_decay': 0.05,
    'mixup_alpha': 0.2
}
```

## 🔧 Advanced Configuration

### Model Architecture Details

#### YOLO Configuration
- **Backbone**: CSPDarknet with Feature Pyramid Network
- **Input Size**: 640×640 pixels
- **Optimization**: AdamW with cosine annealing
- **Transfer Learning**: Pre-trained on COCO dataset

#### EfficientNet Configuration
- **Architecture**: EfficientNet-B0 (5.3M parameters)
- **Custom Head**: FC layers with batch normalization
- **Optimization**: AdamW with differential learning rates
- **Regularization**: Dropout (0.2), Label Smoothing (0.1)

#### ViT Configuration
- **Architecture**: ViT-Base/16 (86M parameters)
- **Patch Size**: 16×16 pixels (196 patches per image)
- **Attention Heads**: 12 multi-head attention
- **Optimization**: AdamW with warmup scheduling

### Dataset Requirements

#### For YOLO Training:
```
stanford_cars_yolo/
├── data/
│   ├── dataset.yaml      # YOLO dataset configuration
│   ├── classes.txt       # Class names (196 car types)
│   ├── train/           # Training images and labels
│   ├── val/             # Validation images and labels
│   └── test/            # Test images and labels
```

#### For Headlight Classification:
```
enhanced_headlight_dataset/
├── train/
│   ├── train_headlight_mapping.csv
│   └── headlight_images/
└── test/
    ├── test_headlight_mapping.csv
    └── headlight_images/
```

**CSV Format:**
```csv
stanford_class_id,stanford_class_name,headlight_path,split
0,"AM General Hummer SUV 2000",train/headlight_images/img_001.jpg,train
1,"Acura RL Sedan 2012",train/headlight_images/img_002.jpg,train
```

## 🔍 Technical Implementation Details

### Headlight Detection Pipeline

The project uses a sophisticated headlight detection pipeline:

1. **Traditional CV Attempts**: Edge detection, Hough circles, brightness segmentation
2. **Fourier Transform Method**: Frequency domain pattern matching
3. **Deep Learning Solution**: Roboflow pre-trained car parts detector

### Model Training Strategies

#### Transfer Learning Approach:
- **Stage 1**: Freeze backbone, train classification head
- **Stage 2**: Unfreeze top layers for fine-tuning
- **Stage 3**: End-to-end training with reduced learning rate

#### Data Augmentation:
- **Geometric**: Random crop, horizontal flip, rotation
- **Photometric**: Color jitter, brightness adjustment
- **Advanced**: Mixup, CutMix (ViT only), Random erasing

#### Optimization Techniques:
- **Gradient Clipping**: Prevents exploding gradients
- **Label Smoothing**: Reduces overconfident predictions
- **Weighted Sampling**: Handles class imbalance
- **Early Stopping**: Prevents overfitting

## 📈 Performance Analysis

### Experimental Results

The project demonstrates significant architectural differences:

| Metric | YOLO | EfficientNet | ViT |
|--------|------|--------------|-----|
| **Input Type** | Full Vehicle | Headlight Only | Headlight Only |
| **Architecture** | CNN + Detection Head | CNN + FC Head | Transformer + FC Head |
| **Parameters** | 3.2M | 5.3M | 86M |
| **Training Time** | 4 hours | 30 hours | 8 hours |
| **Top-1 Accuracy** | 85% | 48% | 85% |
| **Top-5 Accuracy** | 95% | 88% | 95% |

### Key Findings

1. **Transformer Superiority**: ViT significantly outperforms CNN for headlight-based recognition (85% vs 48%)
2. **Attention Mechanism Advantage**: Self-attention captures subtle design patterns effectively
3. **Component vs Full Vehicle**: Headlight-based ViT matches full vehicle YOLO performance
4. **Training Efficiency**: ViT requires less training time than EfficientNet despite larger size

## 🎯 Applications

### Real-World Use Cases

- **Traffic Monitoring**: Automated vehicle identification in surveillance systems
- **Parking Management**: Vehicle recognition in parking lots
- **Security Systems**: Access control based on vehicle identification
- **Insurance Claims**: Automated vehicle model verification
- **Market Research**: Automotive industry analysis

### Deployment Scenarios

#### Full Vehicle Recognition (YOLO):
- Highway monitoring systems
- Toll booth automation
- Traffic flow analysis

#### Headlight-Based Recognition (ViT):
- Partial occlusion scenarios
- Front-facing security cameras
- Component-specific analysis

## 🐛 Troubleshooting

### Common Issues

#### CUDA Memory Issues:
```bash
# Reduce batch size
--batch-size 8

# Use gradient checkpointing (ViT)
use_gradient_checkpointing=True
```

#### Dataset Not Found:
```bash
# Check CSV file paths
enhanced_headlight_dataset/train/train_headlight_mapping.csv
enhanced_headlight_dataset/test/test_headlight_mapping.csv
```

#### Model Loading Errors:
```bash
# Ensure model checkpoint exists
headlight_classifier_output/best_model.pth
vit_headlight_classifier_output/best_vit_model.pth
```

### Performance Optimization

#### For Training:
- Use mixed precision training: `torch.cuda.amp`
- Enable gradient checkpointing for large models
- Use appropriate batch sizes for your GPU memory

#### For Inference:
- Convert models to TorchScript for faster inference
- Use batch processing for multiple images
- Implement model quantization for deployment

## 📚 Research Contributions

This project contributes to the field by:

1. **Novel Classification Paradigm**: Introduction of headlight-based vehicle recognition
2. **Architecture Comparison**: Systematic evaluation of CNN vs Transformer approaches
3. **Practical Insights**: Performance trade-offs for real-world deployment
4. **Benchmark Establishment**: Performance baselines for component-based recognition

## 📄 Citation

If you use this work in your research, please cite:

```bibtex
@misc{polat2024car,
  title={Car Make and Model Recognition: A Comparative Study of YOLO-Based Full Vehicle Detection and Headlight-Based Classification Using EfficientNet and ViT},
  author={Polat, Ömer Faruk},
  year={2024},
  institution={Hacettepe University},
  url={https://github.com/omerfarukpolat/Car-Brand-and-Model-Detector-from-Vehicle-Parts}
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📞 Contact

**Ömer Faruk Polat**  
Department of Computer Engineering  
Hacettepe University, Ankara, Turkey  
📧 omerpolat@hacettepe.edu.tr  
🔗 [GitHub Repository](https://github.com/omerfarukpolat/Car-Brand-and-Model-Detector-from-Vehicle-Parts)

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Keywords**: Computer Vision, Deep Learning, Vehicle Recognition, YOLO, EfficientNet, Vision Transformer, Stanford Cars Dataset, Headlight Classification
