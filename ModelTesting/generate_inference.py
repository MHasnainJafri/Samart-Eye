# generate_inference_csv.py

import os
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import pandas as pd
from tqdm import tqdm

# Set paths
TEST_DIR = "/mnt/d/FinalYear/processed_dataset/finalize_data/test"
MODEL_PATH = "best_model.pth"  # Replace this with your actual model path
OUTPUT_CSV = "inference.csv"

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load test dataset
test_dataset = datasets.ImageFolder(TEST_DIR, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=False)  # Replace with your model arch
model.fc = torch.nn.Linear(model.fc.in_features, len(test_dataset.classes))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

# Run inference
results = []

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Running Inference"):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        probs = F.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)

        for i in range(images.size(0)):
            item = test_loader.dataset.samples[i]
            filename = os.path.basename(item[0])
            true_label = test_dataset.classes[labels[i]]
            predicted = test_dataset.classes[preds[i]]
            prob_dict = {test_dataset.classes[j]: float(probs[i][j]) for j in range(len(test_dataset.classes))}

            results.append({
                "filename": filename,
                "label": true_label,
                "predicted": predicted,
                **prob_dict
            })

# Save to CSV
df = pd.DataFrame(results)
df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved inference results to {OUTPUT_CSV}")
