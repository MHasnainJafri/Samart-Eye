import os
import json
import torch
import numpy as np
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from collections import defaultdict

# ---------- Configuration ----------
DATA_DIR = "/mnt/d/FinalYear/processed_dataset/finalize_data"
TEST_DIR = os.path.join(DATA_DIR, "test")
MODEL_PATH = "best_model.pth"
CLASS_NAMES_PATH = "car_model_classes.json"
OUTPUT_JSON = "test_metrics.json"
BATCH_SIZE = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAMPLE_OUTPUT_DIR = "test_samples"
MAX_SAMPLES_PER_CLASS = 5  # Max correct/incorrect samples to save per class

# Create output directories
os.makedirs(SAMPLE_OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(SAMPLE_OUTPUT_DIR, "correct"), exist_ok=True)
os.makedirs(os.path.join(SAMPLE_OUTPUT_DIR, "incorrect"), exist_ok=True)

# Load class names
with open(CLASS_NAMES_PATH, "r") as f:
    CLASS_NAMES = json.load(f)
    CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}

# ---------- Image Transformation ----------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])

# ---------- Load Dataset ----------
test_dataset = datasets.ImageFolder(TEST_DIR, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ---------- Load Model ----------
model = models.resnet50(weights=None)
num_features = model.fc.in_features
model.fc = torch.nn.Sequential(
    torch.nn.Linear(num_features, 1024),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.5),
    torch.nn.Linear(1024, 512),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.4),
    torch.nn.Linear(512, len(CLASS_NAMES)))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE)['model_state_dict'])
model.to(DEVICE)
model.eval()

# ---------- Statistics Tracking ----------
class_stats = defaultdict(lambda: {
    "total": 0,
    "correct": 0,
    "confidence_sum": 0,
    "confidence_bins": {
        "90-100": 0,
        "80-90": 0,
        "70-80": 0,
        "60-70": 0,
        "50-60": 0,
        "0-50": 0
    },
    "wrong_predictions": defaultdict(int),
    "samples": {"correct": [], "incorrect": []}
})

global_stats = {
    "total_images": 0,
    "total_correct": 0,
    "accuracy": 0,
    "confidence_bins": {
        "90-100": 0,
        "80-90": 0,
        "70-80": 0,
        "60-70": 0,
        "50-60": 0,
        "0-50": 0
    },
    "class_distribution": defaultdict(int),
    "confusion_matrix": np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=int)
}

def annotate_image(img_path, actual, predicted, confidence, is_correct):
    """Annotate image with prediction info and save"""
    try:
        img = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(img)

        # Use a readable font
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()

        # Text to display
        text_lines = [
            f"Actual: {actual}",
            f"Predicted: {predicted}",
            f"Confidence: {confidence:.1%}"
        ]
        line_height = font.getsize("Ag")[1] + 4
        text_width = max(font.getsize(line)[0] for line in text_lines)
        total_height = line_height * len(text_lines)

        padding = 10
        box_coords = (0, 0, text_width + 2 * padding, total_height + 2 * padding)

        # Background box
        draw.rectangle(box_coords, fill=(0, 0, 0))

        # Text color
        color = (0, 255, 0) if is_correct else (255, 0, 0)

        # Draw each line of text
        for i, line in enumerate(text_lines):
            position = (padding, padding + i * line_height)
            draw.text(position, line, fill=color, font=font)

        return img
    except Exception as e:
        print(f"Error annotating image {img_path}: {str(e)}")
        return None


def save_annotated_image(img_path, dest_dir, info):
    """Save annotated image to destination directory"""
    base_name = os.path.basename(img_path)
    dest_path = os.path.join(dest_dir, base_name)
    
    annotated_img = annotate_image(
        img_path,
        info["actual"],
        info["predicted"],
        info["confidence"],
        info["is_correct"]
    )
    
    if annotated_img:
        annotated_img.save(dest_path)
        return dest_path
    return None

def update_stats(actual_class, predicted_class, confidence, img_path, img_idx):
    """Update all statistics tracking"""
    is_correct = (actual_class == predicted_class)
    
    # Update global stats
    global_stats["total_images"] += 1
    global_stats["total_correct"] += int(is_correct)
    global_stats["class_distribution"][actual_class] += 1
    global_stats["confusion_matrix"][CLASS_TO_IDX[actual_class]][CLASS_TO_IDX[predicted_class]] += 1
    
    # Update confidence bins
    def update_bins(stats_dict):
        if confidence >= 0.9:
            stats_dict["confidence_bins"]["90-100"] += 1
        elif confidence >= 0.8:
            stats_dict["confidence_bins"]["80-90"] += 1
        elif confidence >= 0.7:
            stats_dict["confidence_bins"]["70-80"] += 1
        elif confidence >= 0.6:
            stats_dict["confidence_bins"]["60-70"] += 1
        elif confidence >= 0.5:
            stats_dict["confidence_bins"]["50-60"] += 1
        else:
            stats_dict["confidence_bins"]["0-50"] += 1
    
    update_bins(global_stats)
    
    # Update class-specific stats
    stats = class_stats[actual_class]
    stats["total"] += 1
    stats["correct"] += int(is_correct)
    stats["confidence_sum"] += confidence
    update_bins(stats)
    
    if not is_correct:
        stats["wrong_predictions"][predicted_class] += 1
    
    # Save sample images if needed
    sample_type = "correct" if is_correct else "incorrect"
    if len(stats["samples"][sample_type]) < MAX_SAMPLES_PER_CLASS:
        dest_dir = os.path.join(SAMPLE_OUTPUT_DIR, sample_type)
        saved_path = save_annotated_image(
            img_path,
            dest_dir,
            {
                "actual": actual_class,
                "predicted": predicted_class,
                "confidence": confidence,
                "is_correct": is_correct
            }
        )
        if saved_path:
            stats["samples"][sample_type].append({
                "image_path": saved_path,
                "confidence": confidence,
                "predicted_class": predicted_class
            })

# ---------- Run Evaluation ----------
with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Evaluating"):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        outputs = model(images)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        confidences, preds = torch.max(probs, 1)
        
        for i in range(images.size(0)):
            img_idx = global_stats["total_images"]
            img_path, actual_idx = test_dataset.samples[img_idx]
            actual_class = CLASS_NAMES[actual_idx]
            predicted_class = CLASS_NAMES[preds[i].item()]
            confidence = confidences[i].item()
            
            update_stats(
                actual_class,
                predicted_class,
                confidence,
                img_path,
                img_idx
            )

# Calculate final metrics
global_stats["accuracy"] = global_stats["total_correct"] / global_stats["total_images"]

for class_name, stats in class_stats.items():
    stats["accuracy"] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
    stats["avg_confidence"] = stats["confidence_sum"] / stats["total"] if stats["total"] > 0 else 0
    # Convert wrong_predictions defaultdict to regular dict for JSON
    stats["wrong_predictions"] = dict(stats["wrong_predictions"])

# Prepare final output
output = {
    "global_stats": {
        **global_stats,
        "confusion_matrix": global_stats["confusion_matrix"].tolist(),
        "class_distribution": dict(global_stats["class_distribution"])
    },
    "class_stats": {k: dict(v) for k, v in class_stats.items()},
    "class_names": CLASS_NAMES,
    "sample_images_dir": SAMPLE_OUTPUT_DIR
}

# Save JSON
with open(OUTPUT_JSON, "w") as f:
    json.dump(output, f, indent=2)

print(f"\nEvaluation Complete:")
print(f"✅ Total Images: {global_stats['total_images']}")
print(f"✅ Accuracy: {global_stats['accuracy']:.2%}")
print(f"✅ Confidence Distribution:")
for bin_name, count in global_stats['confidence_bins'].items():
    print(f"   - {bin_name}%: {count} ({count/global_stats['total_images']:.1%})")
print(f"✅ Results saved to {OUTPUT_JSON}")
print(f"✅ Sample images saved to {SAMPLE_OUTPUT_DIR}")