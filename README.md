# Sign Language recognition system - based on anatomical constraints and spatial attention mechanisms

## 📌 Project introduction
This project aims to build a sign language recognition system with high accuracy and low latency for barrier-free communication. It innovatively combines anatomical constraints with spatial attention mechanisms to improve the accuracy and robustness of gesture recognition. Real-time inference is realized through optimized deployment of embedded platform.

---

🚀 Core technology and method
### ✅ Anatomical constraints + spatial attention mechanisms
- Introduce ** joint kinematic restriction ** and ** phalangeal length ratio restriction ** to enhance gesture feature representation and ensure biological rationality.
- Spatial attention mechanism optimizes feature weight distribution and improves recognition performance in complex scenes.

### 🔥 Data enhancement and optimization
- Use SMOTE oversampling and SMOTE composite data to enhance the SMOTE strategy to resolve the data deficiency issue.

### ⚙️ embedded platform optimization
- Deploy the model on **Jetson Xavier NX**, optimize compression, achieve **32FPS** real-time inference performance.
- Model volume compression **68%**, improve computing efficiency.

### 🌐 multi-modal interaction module
- Realize real-time conversion of sign language, text and speech to enhance user interaction experience.

---

## 📊 experimental results
- ** Average recognition accuracy: 99.86%** (under complex background), **12.6 percentage points higher than the traditional CNN method (87.26%) ** *.
- Embedded platform achieves the highest real-time inference performance at 32FPS.

---

## 💡 Project highlights
- 🚀 ** End-to-end Fusion ** : The first end-to-end fusion of anatomical constraints and deep learning models.
- ⚡ ** Efficient Inference ** : Real-time inference with high precision and low latency on embedded platforms.
- ♿ ** Barrier-free communication ** : To provide high-precision real-time sign language recognition and translation solutions for special groups.

---

## 🔧 Deployment and operation

### 📥 Environment configuration
1. ** Hardware requirements **
- Jetson Xavier NX
- CUDA 11.4 / cuDNN 8.2
- TensorRT 8.2

2. ** Software environment **
- Python 3.8
- PyTorch >= 1.10
- OpenCV, NumPy, TensorRT

---

### ⚙️ steps
1. ** Cloning project **
```bash
git clone https://github.com/Scatteredpeople/HandFind.git
cd HandFind/src
` ` `
2. File directory You need to create a folder with video files under the text directory
```bash
src/ # Project root directory
├── data/ # Store all data (raw data, processed data, etc.)
│ ├── picture/ # Raw data (No modification)
│ ├── picture_augmented/ # Data enhanced
│ ├── pridict/ # Forecast picture
│ └ -- # There are also some files about image generated files
├── VideoCreatePricture.py # Video generating image
├── DataPictureAugmented.py # Image enhancement
├── FindHandCsv.py # Find the coordinates of the hand joint
├── ─ FindImgnameError.py # Handles image data that identifies errors
├── DelAgmentedPicture.py # Removes enhanced images that are not recognized
├── clean.py # Clean up data
├── TrainModel.py # Training model
├── Pridict.py # Forecast picture
` ` `
