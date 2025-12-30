# Brain-Tumour-Glioma-Classification
Streamlit app deploying a hybrid classical-quantum AI for brain tumor detection. Combines ResNet18 with a 4-qubit quantum circuit (98.57% accuracy). Features batch processing of up to 10 MRI scans, real-time predictions with confidence scores, and clinical-grade interface for medical research.
# 🧠 Brain Tumor Classification - Hybrid Classical-Quantum AI

A practical implementation of a hybrid classical-quantum neural network for detecting glioma tumors in brain MRI scans. This web application combines traditional deep learning (ResNet18) with quantum computing to achieve state-of-the-art classification accuracy.

## ✨ Key Features

- **Hybrid Architecture**: ResNet18 feature extraction + 4-qubit quantum circuit processing
- **High Accuracy**: 98.57% validation accuracy during training
- **Batch Processing**: Analyze up to 10 MRI images simultaneously
- **User-Friendly Interface**: Intuitive Streamlit web app for medical researchers
- **Confidence Metrics**: Transparent probability scores for each prediction

## 🚀 Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/brain-tumor-classifier.git
cd brain-tumor-classifier
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download model weights** (available upon request)
Place `best_model.pth` in the project root directory

4. **Run the application**
```bash
streamlit run app1.py
```

## 🏗️ Technical Architecture

The model implements a "dressed quantum" approach:
- **Classical Backbone**: Pre-trained ResNet18 for feature extraction
- **Quantum Processor**: 4-qubit variational circuit with 6 layers
- **Post-Quantum Classifier**: Fully connected neural network

## 📊 Performance

- **Training Accuracy**: 98.57%
- **Test Loss**: 0.0861
- **Deployment Accuracy**: 99.29% (5,957 images)

## ⚠️ Important Notes

- **For Research Use Only**: This tool is designed for academic research
- **Medical Disclaimer**: Not for clinical diagnosis - always consult healthcare professionals
- **Model Weights**: Available to verified researchers upon request

## 🔗 Links

- [Research Paper] (Link to your thesis/publication)
- [Live Demo] (Streamlit Cloud link if deployed)
- [Dataset Information] (Link to dataset source)

## 📄 License

This project is available for academic and research purposes. See LICENSE file for details.

---

*Built with PyTorch, PennyLane, and Streamlit • Part of MSc/PhD Research in Quantum Machine Learning*
