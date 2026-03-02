"""
Plant Disease Inference Script
Predicts disease class for any leaf image using the trained model.
"""

import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MODEL_PATH = r"C:\Users\goray\Desktop\FYP\PDS\plant_disease_model.pth"
IMG_SIZE = 224

# Classes are loaded dynamically from checkpoint
# See idx_to_class after loading model


def load_model(model_path: str, device: torch.device):
    """Load the trained model with its checkpoint."""
    checkpoint = torch.load(model_path, map_location=device)

    # Get number of classes from the checkpoint
    class_to_idx = checkpoint.get('class_to_idx', {})
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(class_to_idx)

    print(f"Loaded model from: {model_path}")
    print(f"Number of classes: {num_classes}")
    print(f"Best validation accuracy: {checkpoint.get('val_acc', 'N/A')}%")
    print(f"Epoch: {checkpoint.get('epoch', 'N/A')}")

    # Build model (same architecture as training)
    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(model.fc.in_features, num_classes)
    )

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model, idx_to_class


def get_transform():
    """Get the same transforms used during validation."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def predict_image(model, image_path: str, device: torch.device, idx_to_class: dict):
    """
    Predict the disease class for a single leaf image.

    Args:
        model: Trained PyTorch model
        image_path: Path to the leaf image
        device: CUDA or CPU device
        idx_to_class: Mapping from index to class name

    Returns:
        Predicted class name and confidence scores
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    transform = get_transform()
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)

    predicted_class = idx_to_class[predicted_idx.item()]
    confidence_score = confidence.item()

    # Get top 3 predictions
    top_probs, top_indices = torch.topk(probabilities, 3, dim=1)
    top_predictions = [
        (idx_to_class[idx.item()], prob.item())
        for idx, prob in zip(top_indices[0], top_probs[0])
    ]

    return predicted_class, confidence_score, top_predictions


def predict_batch(model, image_folder: str, device: torch.device, idx_to_class: dict, max_images: int = 10):
    """
    Predict disease classes for all images in a folder.

    Args:
        model: Trained PyTorch model
        image_folder: Path to folder containing leaf images
        device: CUDA or CPU device
        idx_to_class: Mapping from index to class name
        max_images: Maximum number of images to predict (0 for all)

    Returns:
        List of predictions
    """
    if not os.path.isdir(image_folder):
        raise NotADirectoryError(f"Folder not found: {image_folder}")

    # Get all image files
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    image_files = [
        f for f in os.listdir(image_folder)
        if os.path.splitext(f.lower())[1] in valid_extensions
    ]

    if max_images > 0:
        image_files = image_files[:max_images]

    print(f"\nFound {len(image_files)} images. Predicting...")
    print("-" * 60)

    results = []
    transform = get_transform()

    for img_file in image_files:
        img_path = os.path.join(image_folder, img_file)

        try:
            image = Image.open(img_path).convert('RGB')
            image_tensor = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)

            predicted_class = idx_to_class[predicted_idx.item()]
            confidence_score = confidence.item()

            results.append({
                'file': img_file,
                'prediction': predicted_class,
                'confidence': confidence_score
            })

            print(f"{img_file}: {predicted_class} ({confidence_score*100:.1f}%)")

        except Exception as e:
            print(f"Error processing {img_file}: {e}")

    return results


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model, idx_to_class = load_model(MODEL_PATH, device)

    # ============================================
    # OPTION 1: Test a single leaf image
    # ============================================
    # Replace with your image path here
    test_image_path = r"C:\Users\goray\Desktop\FYP\Data-kaggle\New Plant Diseases Dataset(Augmented)\valid\Tomato___Late_blight\005a2c1f-4e15-49e4-9e5c-61dc3ecf9708___RS_Late.B 5096.JPG"

    # Test single image:
    predicted_class, confidence, top_predictions = predict_image(model, test_image_path, device, idx_to_class)
    print(f"\nPrediction: {predicted_class}")
    print(f"Confidence: {confidence*100:.2f}%")
    print("\nTop 3 predictions:")
    for cls, prob in top_predictions:
        print(f"  {cls}: {prob*100:.1f}%")

    print("\n" + "="*60)
    print("Inference complete!")
    print("="*60)
