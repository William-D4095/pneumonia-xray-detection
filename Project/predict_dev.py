import argparse
import numpy as np
import onnxruntime as ort
from PIL import Image

# PyTorch ImageFolder assigns classes alphabetically: 0 -> NORMAL, 1 -> PNEUMONIA
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

def preprocess_image(image_path):
    """Loads, resizes, and normalizes an input image to match ResNet-18 expectations."""
    # 1. Load image and convert to 3-channel RGB
    img = Image.open(image_path).convert('RGB')
    
    # 2. Resize to the target 224x224 input size
    img = img.resize((224, 224))
    
    # 3. Convert to float array and scale pixels to [0, 1]
    img_data = np.array(img, dtype=np.float32) / 255.0
    
    # 4. Normalize with ImageNet mean and standard deviation
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_data = (img_data - mean) / std
    
    # 5. Transpose layout from HWC (224, 224, 3) to CHW (3, 224, 224)
    img_data = np.transpose(img_data, (2, 0, 1))
    
    # 6. Add batch dimension -> (1, 3, 224, 224)
    img_data = np.expand_dims(img_data, axis=0)
    
    return img_data

def run_inference(model_path, image_path):
    # Preprocess the input image
    input_data = preprocess_image(image_path)
    
    # Set up ONNX Runtime session using CUDA GPU acceleration first, falling back to CPU
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(model_path, providers=providers)
    
    # Fetch input and output tensor names from the ONNX graph
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    # Execute model inference
    outputs = session.run([output_name], {input_name: input_data})
    logits = outputs[0][0]
    
    # Compute class probabilities using Softmax
    exp_logits = np.exp(logits - np.max(logits))
    probabilities = exp_logits / np.sum(exp_logits)
    
    # Get predicted class index and confidence score
    predicted_idx = int(np.argmax(probabilities))
    predicted_label = CLASS_NAMES[predicted_idx]
    confidence = probabilities[predicted_idx] * 100
    
    print(f"--- Inference Result ---")
    print(f"Predicted Class : {predicted_label} ({confidence:.2f}% confidence)")
    print(f"Class Scores    : NORMAL = {probabilities[0]*100:.2f}%, PNEUMONIA = {probabilities[1]*100:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run pneumonia ONNX classification on Jetson Orin Nano.")
    parser.add_argument("--model", type=str, default="pneumonia_resnet18_model.onnx", help="Path to ONNX model file")
    parser.add_argument("--image", type=str, required=True, help="Path to input chest X-ray image")
    args = parser.parse_args()
    
    run_inference(args.model, args.image)
