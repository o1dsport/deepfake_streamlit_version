# Deepfake AI Detection System

An AI-powered deepfake detection web application built using TensorFlow, MobileNetV2, and Streamlit.  
This project focuses on building a practical deepfake classifier and understanding real-world machine learning deployment challenges.

---

## Live Demo

https://deepfakeappversion.streamlit.app

## Project Overview

Deepfake content is becoming increasingly realistic and difficult to detect.  
This system classifies an uploaded facial image as **Real** or **Fake** and provides a confidence score.

To improve transparency, the model also supports Grad-CAM visualization to highlight regions influencing the prediction.

---

## Model Architecture

- **Base Model:** MobileNetV2 (pretrained on ImageNet)  
- **Approach:** Transfer Learning  
- **Classification:** Binary (Sigmoid activation)  
- **Image Size:** 224 × 224  
- **Optimizer:** Adam  
- **Loss Function:** Binary Crossentropy  
- Data augmentation used during training  
- EarlyStopping and ModelCheckpoint callbacks  

---

## Tech Stack

**Backend**
- Python  
- TensorFlow  
- NumPy  
- OpenCV (Headless)  

**Frontend**
- Streamlit  

---

## How to Run Locally

1. Clone the repository:
git clone https://github.com/o1dsport/deepfake-ai-detection.git
cd deepfake-ai-detection


2. Install dependencies:
pip install -r requirements.txt


3. Place the trained model file in the root directory.

4. Run the application:
streamlit run app.py


5. Open in browser:
http://localhost:8501


---

## Deployment Experience & Engineering Challenges

While attempting deployment to cloud platforms, several real-world ML challenges were encountered:

- TensorFlow increases build size significantly
- OpenCV system dependencies caused runtime errors in headless environments
- Model serialization compatibility issues across TensorFlow versions
- Environment version mismatches (TensorFlow, NumPy, h5py)
- Production port binding and server configuration requirements

Although full deployment was not finalized, the process provided valuable insight into ML system engineering beyond model training.

---

## What This Project Demonstrates

- End-to-end ML pipeline development  
- Transfer learning implementation  
- Explainable AI integration  
- ML + Web application integration  
- Understanding of deployment constraints  

---

## Future Improvements

- Docker containerization  
- Lightweight model export (ONNX)  
- CI/CD pipeline integration  
- Cloud-native deployment  
- Video deepfake detection  

---

## Author

Ashish Kumar  
B.Tech – AI & Machine Learning  
