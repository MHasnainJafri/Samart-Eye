import os
import json
import torch
import numpy as np
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import shutil
from collections import defaultdict

# ---------- Configuration ----------
DATA_DIR = "/mnt/d/FinalYear/processed_dataset/finalize_data"
TEST_DIR = os.path.join(DATA_DIR, "test")
MODEL_PATH = "best_model.pth"
CLASS_NAMES_PATH = "car_model_classes.json"
OUTPUT_JSON = "test_metrics.json"
BATCH_SIZE = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAMPLE_OUTPUT_DIR = "test_samples"  # Directory to save sample images

# Create output directories
os.makedirs(SAMPLE_OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(SAMPLE_OUTPUT_DIR, "correct"), exist_ok=True)
os.makedirs(os.path.join(SAMPLE_OUTPUT_DIR, "incorrect"), exist_ok=True)

# ---------- Load Class Names ----------
with open(CLASS_NAMES_PATH, "r") as f:
    CLASS_NAMES = json.load(f)

# ---------- Transform for test images ----------
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
    torch.nn.Linear(512, len(CLASS_NAMES))
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE)['model_state_dict'])
model.to(DEVICE)
model.eval()

# ---------- Prediction and Statistics ----------
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
    "sample_images": {
        "correct": [],
        "incorrect": []
    }
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
    "confusion_matrix": np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)))
}

def save_sample_image(src_path, dest_dir, info):
    """Save a sample image with prediction info"""
    base_name = os.path.basename(src_path)
    dest_path = os.path.join(dest_dir, base_name)
    
    # Copy original image
    shutil.copy2(src_path, dest_path)
    
    # Create annotation file
    with open(os.path.join(dest_dir, f"{os.path.splitext(base_name)[0]}.txt"), "w") as f:
        f.write(json.dumps(info, indent=2))

def update_confidence_bins(stats_dict, confidence):
    """Update confidence bin counts"""
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

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Testing"):
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
            
            # Update global statistics
            global_stats["total_images"] += 1
            global_stats["class_distribution"][actual_class] += 1
            global_stats["confusion_matrix"][actual_idx][preds[i].item()] += 1
            update_confidence_bins(global_stats, confidence)
            
            # Update class-specific statistics
            stats = class_stats[actual_class]
            stats["total"] += 1
            stats["confidence_sum"] += confidence
            update_confidence_bins(stats, confidence)
            
            is_correct = (actual_idx == preds[i].item())
            if is_correct:
                stats["correct"] += 1
                global_stats["total_correct"] += 1
                
                # Save sample correct prediction (limit to 5 per class)
                if len(stats["sample_images"]["correct"]) < 5:
                    save_sample_image(
                        img_path,
                        os.path.join(SAMPLE_OUTPUT_DIR, "correct"),
                        {
                            "actual": actual_class,
                            "predicted": predicted_class,
                            "confidence": confidence,
                            "is_correct": True
                        }
                    )
                    stats["sample_images"]["correct"].append(os.path.basename(img_path))
            else:
                stats["wrong_predictions"][predicted_class] += 1
                
                # Save sample incorrect prediction (limit to 5 per class)
                if len(stats["sample_images"]["incorrect"]) < 5:
                    save_sample_image(
                        img_path,
                        os.path.join(SAMPLE_OUTPUT_DIR, "incorrect"),
                        {
                            "actual": actual_class,
                            "predicted": predicted_class,
                            "confidence": confidence,
                            "is_correct": False
                        }
                    )
                    stats["sample_images"]["incorrect"].append(os.path.basename(img_path))

# Calculate final metrics
global_stats["accuracy"] = global_stats["total_correct"] / global_stats["total_images"]

for class_name, stats in class_stats.items():
    stats["accuracy"] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
    stats["avg_confidence"] = stats["confidence_sum"] / stats["total"] if stats["total"] > 0 else 0

# Convert confusion matrix to list for JSON serialization
global_stats["confusion_matrix"] = global_stats["confusion_matrix"].tolist()

# Prepare final output
output = {
    "global_stats": global_stats,
    "class_stats": class_stats,
    "class_names": CLASS_NAMES,
    "sample_images_dir": SAMPLE_OUTPUT_DIR
}

# Save JSON
with open(OUTPUT_JSON, "w") as f:
    json.dump(output, f, indent=2)

print(f"✅ Saved comprehensive test metrics to {OUTPUT_JSON}")
print(f"✅ Sample images saved to {SAMPLE_OUTPUT_DIR}")
print(f"\nGlobal Accuracy: {global_stats['accuracy']:.2%}")
print(f"Total Images Tested: {global_stats['total_images']}")