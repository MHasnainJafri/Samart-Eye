# Vehicle Detection and Tracking System: Technical Documentation

## 1. Introduction

### 1.1 Purpose
This system integrates the CARLA simulator, YOLOv11, and ResNet50 to enable real-time detection, classification, and tracking of specific vehicle types (e.g., Dodge Charger Police) in urban environments. The system provides visualization through a Pygame-based dashboard for monitoring and analysis.

### 1.2 Scope
The system processes multi-camera streams to detect, classify, and track vehicles, with the following specifications:
- **Input**: Multi-camera RGB streams from CARLA (800x600 resolution, 20 FPS).
- **Processing**:
  - Region of Interest (ROI)-based vehicle detection using YOLOv11.
  - Fine-grained vehicle classification using ResNet50.
  - Priority tracking of target vehicles with blinking alerts.
- **Output**:
  - Pygame dashboard displaying a 3x2 camera grid.
  - Recorded videos with detection overlays.
  - JSON logs capturing ROIs and detection metadata.

### 1.3 Key Technologies
| Component       | Version         |
|-----------------|-----------------|
| Python          | 3.10            |
| PyTorch         | 2.7.1+cu118     |
| CARLA           | 0.9.14          |
| YOLOv11         | Ultralytics     |
| OpenCV          | 4.12.0.88       |

## 2. System Architecture


### 2.1 End-to-End Workflow
![End-to-End Workflow Diagram](https://raw.githubusercontent.com/mhj-dev/giki/273181ae216f8d39a92e818068e0beccb131def7/compoennet%20diagream.png)
- **CARLA Image Capture** → **Preprocessing** → **Model Training** → **Pygame Visualization**

### 2.2 Detailed Data Flow
![Detailed Data Flow Diagram](https://raw.githubusercontent.com/mhj-dev/giki/273181ae216f8d39a92e818068e0beccb131def7/workflow.png)
- **CARLA Simulator** → **Camera Feeds** → **YOLOv11 Detector** → **ResNet50 Classifier** → **Pygame Dashboard**, **ROI JSON Logs**, **Video Recordings**

### 2.3 Preprocessing 
![Preprocessing](https://raw.githubusercontent.com/mhj-dev/giki/main/Your%20paragraph%20text.jpg)






## 3. Detailed Requirements

### 3.1 Data Pipeline

#### 3.1.1 Data Collection (CARLA Simulator)
| ID  | Requirement                              | Technical Implementation                                      |
|-----|------------------------------------------|-------------------------------------------------------------|
| DC1 | Capture 500+ images per vehicle model    | CARLA `sensor.camera.rgb` with randomized parameters: <br> - Weather conditions <br> - Camera angles <br> - Lighting variations |

#### 3.1.2 Preprocessing
| Step                    | Tools/Methods                          | Output                     |
|-------------------------|----------------------------------------|----------------------------|
| Corrupted Image Removal | PIL.Image.verify()                     | Clean dataset              |
| Vehicle Cropping        | YOLOv11 + 10% padding                  | 224x224 image crops        |
| Augmentation            | OpenCV/PIL: Flips, Noise, CLAHE        | 6x dataset size            |

**Augmentation Transformations**:
```python
transformations = [
    ("original", lambda x: x),
    ("flip", lambda x: cv2.flip(x, 1)),
    ("brightness", lambda x: cv2.convertScaleAbs(x, alpha=1.2, beta=20)),
    ("contrast", lambda x: cv2.convertScaleAbs(x, alpha=1.5, beta=0)),
    ("hue", lambda x: cv2.cvtColor(np.clip(cv2.cvtColor(x, cv2.COLOR_BGR2HSV).astype(np.int32) + np.array([20, 0, 0], dtype=np.int32), 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)),
    ("saturation", lambda x: cv2.cvtColor(np.clip(cv2.cvtColor(x, cv2.COLOR_BGR2HSV).astype(np.float32) * np.array([1, 1.3, 1], dtype=np.float32), 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)),
    ("equalized", apply_color_histogram_equalization),
    ("noisy", add_noise)
]
```

### 3.2 Model Training

#### 3.2.1 Architecture
| Component       | Specification                                      |
|-----------------|---------------------------------------------------|
| Base Model      | ResNet50 (pre-trained on ImageNet)                |
| Custom Head     | 1024 → 512 → N_classes (Dropout 0.5)              |
| Loss Function   | CrossEntropyLoss (label_smoothing=0.15)           |

#### 3.2.2 Training Protocol
| Parameter            | Value                                    |
|----------------------|------------------------------------------|
| Batch Size           | 8                                        |
| Optimizer            | Adam (lr=3e-5, weight_decay=1e-5)        |
| Early Stopping       | Patience=15 epochs                       |
| Augmentation         | RandomResizedCrop, ColorJitter           |
| Validation Target    | ≥90% accuracy on CARLA test set          |

#### 3.2.3 ROC Curve

![Roc Curve](https://raw.githubusercontent.com/mhj-dev/giki/main/roc_curv.png)


### 3.3 Simulation Testing

#### 3.3.1 CARLA Integration
| Feature             | Implementation                              |
|---------------------|---------------------------------------------|
| Camera Grid         | 3x2 Pygame display (1200x800 resolution)    |
| ROI Management      | JSON-persisted regions of interest         |
| Target Highlighting | Blinking red border (10-frame interval)    |

#### 3.3.2 Performance Metrics
| Metric                | Target                     |
|-----------------------|----------------------------|
| Detection FPS         | ≥23 FPS (with YOLOv11)    |

### 3.4 Limitations
| Challenge             | Mitigation Strategy                     |
|-----------------------|-----------------------------------------|
| Domain Gap (Sim→Real) | Fine-tune on 10% real-world data        |
| Variable Lighting     | CLAHE preprocessing                     |

## 4. Future Roadmap
1. **Phase 1 (Current)**: Prototype in CARLA simulator.
2. **Phase 2**: Hybrid training with CARLA and real-world dashcam data.

3. **Phase 3**: Deployment on NVIDIA Jetson for edge computing.


#### 4 Project screeenshots 


![Screen shot 1](https://raw.githubusercontent.com/mhj-dev/giki/main/s1.png)
![Screen shot 1](https://raw.githubusercontent.com/mhj-dev/giki/main/s2.png)
![Screen shot 1](https://raw.githubusercontent.com/mhj-dev/giki/main/s3.png)
![Screen shot 1](https://raw.githubusercontent.com/mhj-dev/giki/main/s4.png)



## 5. Material Required for the Project

- **Model**: [Download Model](https://drive.google.com/file/d/10yWf5O79NOMYm6EA4wEnZyNLJy6M5iSZ/view?usp=sharing)  
- **Raw Data**: [Download Raw Data](https://drive.google.com/file/d/14r3ipqJrCjTre6uMkzvr9088EwLcLBPN/view?usp=sharing)
