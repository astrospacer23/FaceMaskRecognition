Face Mask Detection System (PyTorch)
A high-performance, real-time face mask detection system built using Python, PyTorch, and MobileNetV2. This project leverages transfer learning and fine-tuning to achieve high accuracy even with smaller datasets.

**Features**
Real-time Detection: Process webcam feeds at high FPS.

Deep Learning Backbone: Uses MobileNetV2 for an optimal balance between speed and accuracy.

Smart Face Detection: Supports both MediaPipe (Modern/High Accuracy) and Haar Cascades (Legacy/High Compatibility).

Advanced Training: Includes data augmentation, learning rate scheduling, and label smoothing.

**Installation**
1. Clone the Repository
Bash
git clone https://github.com/YOUR_USERNAME/FaceMaskDetection.git
cd FaceMaskDetection
2. Install Dependencies
Ensure you have Python 3.8+ installed. Use the following command to install required libraries:

**Bash**
pip install torch torchvision opencv-python matplotlib tqdm mediapipe pillow
Dataset Structure
Organize your images in the following format to ensure the ImageFolder class loads them correctly:

Plaintext
dataset/
├── with_mask/
│   ├── image1.jpg
│   └── image2.png
└── without_mask/
    ├── image3.jpg
    └── image4.png
**Training the Model**
To train the model on your custom dataset, run:

Bash
python train_model.py
The script uses Transfer Learning (freezing early layers of MobileNetV2).

It saves the best-performing model as mask_detector_v3.pth.

A training plot (training_results.png) is generated automatically to visualize loss and accuracy.

**Running Detection**
Once trained, start the real-time webcam detector:

Bash
python detect_mask.py
Green Box: Mask Detected ✅

Red Box: No Mask! ❌

Controls: Press Q to exit the video stream.

**Troubleshooting**
Interpreter Mismatch: > If you receive ModuleNotFoundError despite installing libraries, ensure your terminal is using the same Python environment as your VS Code. Run: python -m pip install [library_name] or use the full path to your python.exe.

MediaPipe Conflict: If you encounter an AttributeError regarding mediapipe.solutions, ensure you do not have a file named mediapipe.py in your project folder, as this interferes with the library import.

📜 License
