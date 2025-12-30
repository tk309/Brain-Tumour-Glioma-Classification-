# streamlit_app.py

import streamlit as st
import time
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import copy

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, transforms

# Pennylane
import pennylane as qml
from pennylane import numpy as np

# Plotting
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

# Set page configuration
st.set_page_config(
    page_title="Brain Tumor Classification",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        background-color: #f0f2f6;
    }
    .confidence-bar {
        height: 20px;
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4);
        border-radius: 10px;
        margin: 5px 0;
    }
    .model-info {
        background-color: #e8f4fd;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .image-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 20px;
        margin: 20px 0;
    }
    .result-card {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        background: white;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">🧠 BRAIN TUMOUR CLASSIFICATION</h1>', unsafe_allow_html=True)

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define the quantum components (EXACTLY as in training code)
@st.cache_resource
def setup_quantum_components():
    n_qubits = 4
    q_depth = 6
    q_delta = 0.005
    
    dev = qml.device("default.qubit", wires=n_qubits)
    
    @qml.qnode(dev, interface="torch")
    def quantum_net(q_input_features, q_weights):
        qml.AngleEmbedding(q_input_features, wires=range(n_qubits))
        qml.StronglyEntanglingLayers(q_weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(position)) for position in range(n_qubits)]
    
    return quantum_net, n_qubits, q_depth, q_delta, dev

# Define the DressedQuantumNet class (EXACTLY as in training code)
class DressedQuantumNet(nn.Module):
    def __init__(self, n_qubits=4, q_depth=6, q_delta=0.005):
        super().__init__()
        self.pre_net = nn.Sequential(
            nn.Linear(512, 128), nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(128, n_qubits))
        self.q_params = nn.Parameter(q_delta * torch.randn(q_depth, n_qubits, 3))
        self.post_net = nn.Sequential(
            nn.Linear(n_qubits, 64), nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(64, 32), nn.GELU(), 
            nn.Dropout(0.2),
            nn.Linear(32, 2))

    def forward(self, input_features):
        input_features = input_features.to(device)
        pre_out = self.pre_net(input_features)
        q_in = torch.tanh(pre_out) * np.pi / 2.0

        # Apply quantum circuit
        quantum_net, _, _, _, dev = setup_quantum_components()
        
        @qml.qnode(dev, interface="torch")
        def quantum_circuit(features, weights):
            qml.AngleEmbedding(features, wires=range(4))
            qml.StronglyEntanglingLayers(weights, wires=range(4))
            return [qml.expval(qml.PauliZ(i)) for i in range(4)]
        
        q_out = torch.stack([
            torch.hstack(quantum_circuit(elem, self.q_params)) 
            for elem in q_in
        ]).float()

        return self.post_net(q_out)

# Load model function - FIXED VERSION
@st.cache_resource
def load_model():
    try:
        # First, create the exact same model architecture as during training
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        model = torchvision.models.resnet18(weights=weights)
        
        # Freeze layers exactly as in training
        for name, param in model.named_parameters():
            if "layer4" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        # Replace final layer with the exact same quantum net class
        model.fc = DressedQuantumNet()
        model = model.to(device)
        
        # Load trained weights
        if os.path.exists('best_model.pth'):
            # Load the state dict
            state_dict = torch.load('best_model.pth', map_location=device)
            
            # Load the state dict
            model.load_state_dict(state_dict)
            model.eval()
            
            return model
        else:
            st.error("Model file 'best_model.pth' not found. Please ensure it's in the same directory.")
            return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Image preprocessing (EXACTLY as in validation)
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)

# Batch prediction function for multiple images
def predict_images(model, image_tensors):
    with torch.no_grad():
        # Stack all image tensors into a batch
        batch_tensor = torch.cat(image_tensors, dim=0)
        outputs = model(batch_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidences, predictions = torch.max(probabilities, 1)
        
        return (predictions.cpu().numpy(), 
                confidences.cpu().numpy(), 
                probabilities.cpu().numpy())

# File validation function - UPDATED TO 10 IMAGES
def validate_images(uploaded_files):
    """Validate uploaded images"""
    if not uploaded_files:
        return False, "No files uploaded"
    
    # CHANGED: Increased limit from 5 to 10
    if len(uploaded_files) > 10:
        return False, f"Too many images uploaded ({len(uploaded_files)}). Maximum allowed is 10."
    
    allowed_types = ['image/jpeg', 'image/jpg', 'image/png', 'image/bmp']
    invalid_files = []
    
    for file in uploaded_files:
        if file.type not in allowed_types:
            invalid_files.append(file.name)
        if file.size > 10 * 1024 * 1024:  # 10MB limit
            invalid_files.append(f"{file.name} (too large)")
    
    if invalid_files:
        return False, f"Invalid files: {', '.join(invalid_files)}"
    
    return True, "All files are valid"

# Main app
def main():
    # Sidebar for model information
    with st.sidebar:
        st.header("MODEL INFORMATION")
        st.markdown("""
        **ARCHITECTURE:**
        - ResNet18 (classical backbone)
        - Quantum variational circuit (4 qubits)
        - Fully connected layers
        
        **TRAINING RESULTS:**
        - Best Accuracy => 98.57%
        - Best Loss => 0.0861
        
        **EXPECTED CLASSES:**
        - glioma_tumor
        - no_tumor
        """)
        
        st.markdown("---")
        st.header("ABOUT")
        st.markdown("""
        This hybrid model demonstrates the potential of 
        quantum machine learning for medical image analysis.
        
        The application uses a **Hybrid Classical-Quantum Neural Network** to classify brain MRI images as either **Glioma Tumour** or **No Tumour**. 
        The model combines **ResNet18** with a quantum circuit for enhanced feature processing.
        
        **Built with:**
        - PyTorch
        - PennyLane
        - Streamlit
        """)
        
        # Model loading status
        st.header("SYSTEM STATUS")
        if torch.cuda.is_available():
            st.success("✅ GPU Acceleration Enabled")
        else:
            st.info("🖥️ **Device:** CPU")

    # Main content
    st.markdown("---")
    st.header("🚀 Model Deployment")
    
    with st.spinner("Loading quantum-classical model..."):
        model = load_model()
    
    if model is None:
        st.error("❌ Failed to load model. Please check if 'best_model.pth' exists.")
        return
    
    st.success("✅ Model loaded successfully!")
    
    # Image upload section
    st.markdown("---")
    st.header("📤 Upload MRI Images")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        uploaded_files = st.file_uploader(
            # CHANGED: Updated text from "MAX 5" to "MAX 10"
            "Choose brain MRI images (MAX 10) 👇", 
            type=['jpg', 'jpeg', 'png', 'bmp'],
            accept_multiple_files=True,
            help="Upload up to 10 brain MRI scans for tumor classification"  # CHANGED: 5 to 10
        )
        
        st.markdown("""
        **Supported Formats:** JPG, JPEG, PNG, BMP """)
        st.markdown("""
        **Maximum Input:** 10
        
        **Expected Input:** Brain Magnetic Resonance Image scans
        """)
        st.markdown("""            
        **NOTE:** The model expects images similar to the training data.
        """)
        st.markdown("""
        **DISCLAIMER**:
        - This tool is for research purposes only. 
        - Always consult healthcare professionals for medical diagnoses.
        """)
    
    with col2:
        st.image("https://images.unsplash.com/photo-1559757148-5c350d0d3c56?w=500", 
                caption="Tumor Classification Tool", use_column_width=True)
    
    # Display and process uploaded images
    if uploaded_files:
        # Validate uploaded files
        is_valid, validation_msg = validate_images(uploaded_files)
        
        if not is_valid:
            st.error(f"❌ {validation_msg}")
            return
        
        st.success(f"✅ {len(uploaded_files)} image(s) uploaded successfully!")
        
        try:
            # Process all images
            images = []
            image_tensors = []
            filenames = []
            
            with st.spinner("Processing images..."):
                for uploaded_file in uploaded_files:
                    image = Image.open(uploaded_file).convert('RGB')
                    images.append(image)
                    filenames.append(uploaded_file.name)
                    image_tensors.append(preprocess_image(image).to(device))
            
            # Display uploaded images in a grid
            st.subheader("📁 UPLOADED IMAGES")
            
            # Use more columns for better display with up to 10 images
            num_cols = min(4, len(images))  # Increased from 3 to 4 columns for better layout
            cols = st.columns(num_cols)
            
            for idx, (image, filename) in enumerate(zip(images, filenames)):
                with cols[idx % num_cols]:
                    st.image(image, caption=filename, use_column_width=True)
            
            # Perform batch prediction
            st.subheader("🔍 ANALYSIS RESULTS")
            
            with st.spinner("Analyzing images with quantum-classical model..."):
                predictions, confidences, all_probabilities = predict_images(model, image_tensors)
            
            # Class names
            class_names = ['glioma_tumor', 'no_tumor']
            
            # Display results for each image
            for idx, (image, filename, prediction, confidence, probabilities) in enumerate(
                zip(images, filenames, predictions, confidences, all_probabilities)
            ):
                with st.container():
                    st.markdown(f"### 📄 Image {idx + 1}: {filename}")
                    
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.image(image, use_column_width=True)
                    
                    with col2:
                        result = class_names[prediction]
                        
                        # Display prediction
                        if prediction == 0:  # glioma_tumor
                            st.error(f"## 🚨 Prediction: {result}")
                            st.markdown("""
                            ⚠️ **Clinical Note:** This prediction indicates potential abnormalities. 
                            Please consult with a medical professional for proper diagnosis.
                            """)
                        else:  # no_tumor
                            st.success(f"## ✅ Prediction: {result}")
                            st.markdown("""
                            💡 **NOTE:** This model's assessment suggests no tumor detected. 
                            Always follow up with qualified healthcare providers.
                            """)
                        
                        # Confidence metrics
                        st.subheader("📊 Confidence Levels")
                        
                        for i, (class_name, prob) in enumerate(zip(class_names, probabilities)):
                            col_prob, col_bar, col_perc = st.columns([2, 3, 1])
                            
                            with col_prob:
                                st.write(f"{class_name}:")
                            
                            with col_bar:
                                st.progress(float(prob))
                            
                            with col_perc:
                                st.write(f"{prob*100:.1f}%")
                        
                        st.write(f"**Overall Confidence:** {confidence*100:.2f}%")
                    
                    st.markdown("---")
            
            # Summary Statistics
            st.subheader("📈 BATCH SUMMARY")
            
            tumor_count = np.sum(predictions == 0)
            no_tumor_count = np.sum(predictions == 1)
            avg_confidence = np.mean(confidences) * 100
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Glioma Tumor", tumor_count)
            with col2:
                st.metric("No Tumor", no_tumor_count)
            with col3:
                st.metric("Average Confidence", f"{avg_confidence:.2f}%")
            
            # Additional summary for larger batches
            if len(uploaded_files) > 5:
                st.info(f"📊 **Batch Analysis Complete**: Processed {len(uploaded_files)} images with {tumor_count} tumor cases and {no_tumor_count} normal cases.")
            
            # Model interpretation
            st.markdown("---")
            st.subheader("🤖 MODEL INTERPRETATION")
            
            interpretation_text = """
            **How the hybrid model works:**
            1. **Classical Processing:** ResNet18 extracts spatial features from the MRI
            2. **Quantum Encoding:** Features are mapped to quantum state rotations
            3. **Quantum Processing:** Variational quantum circuit processes information
            4. **Classical Readout:** Quantum measurements are interpreted for classification
            
            **Batch Processing:** All images are processed efficiently in a single batch for faster analysis.
            """
            st.markdown(interpretation_text)
        
        except Exception as e:
            st.error(f"Error processing images: {str(e)}")
            st.info("Please try with different images or check file integrity.")
    
    else:
        # Demo section when no images are uploaded
        st.markdown("---")
        st.header("🎯 **HOW TO USE**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            ### 1. Upload Images
            Click 'Browse files' and select up to 10 brain MRI images.  # CHANGED: 5 to 10
            Supported formats: JPG, PNG, BMP
            """)
        
        with col2:
            st.markdown("""
            ### 2. Batch Analysis
            The quantum-classical model will process all images efficiently in one batch.
            """)
        
        with col3:
            st.markdown("""
            ### 3. Get Results
            View individual predictions with confidence scores and batch summary statistics.
            """)
        
        # Example batch processing info
        st.markdown("---")
        st.info("""
        **💡 Batch Processing Advantage:** 
        When multiple images are uploaded, the model processes them together for faster analysis 
        while maintaining individual prediction accuracy for each image.
        
        **📈 Increased Capacity:** Supports up to 10 images per batch!
        """)

if __name__ == "__main__":
    main()