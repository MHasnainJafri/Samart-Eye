import os
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

# Configuration
TEST_RESULTS_JSON = "test_metrics.json"
CLASS_NAMES_PATH = "car_model_classes.json"
OUTPUT_DIR = "evaluation_metrics"
PLOT_FORMAT = "html"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load class names
with open(CLASS_NAMES_PATH) as f:
    CLASS_NAMES = json.load(f)
NUM_CLASSES = len(CLASS_NAMES)

# Load test results
with open(TEST_RESULTS_JSON) as f:
    test_results = json.load(f)

# Prepare data for ROC analysis
# Since all predictions are correct (accuracy=1.0), we'll simulate some variance
# by using the confidence scores from your data

# Create synthetic predictions with some errors for ROC analysis
# (since your actual results show 100% accuracy)
y_true = []
y_scores = []
confidence_levels = []

for class_name, stats in test_results['class_stats'].items():
    class_idx = CLASS_NAMES.index(class_name)
    total_samples = stats['total']
    
    # Create true labels (all correct)
    y_true.extend([class_idx] * total_samples)
    
    # Create confidence scores based on your bins distribution
    bins = stats['confidence_bins']
    confidences = []
    
    # 90-100% confidence
    confidences.extend(np.random.uniform(0.9, 1.0, bins['90-100']))
    # 80-90% confidence
    confidences.extend(np.random.uniform(0.8, 0.9, bins['80-90']))
    # 70-80% confidence
    confidences.extend(np.random.uniform(0.7, 0.8, bins['70-80']))
    # 60-70% confidence
    confidences.extend(np.random.uniform(0.6, 0.7, bins['60-70']))
    # 50-60% confidence
    confidences.extend(np.random.uniform(0.5, 0.6, bins['50-60']))
    # 0-50% confidence
    confidences.extend(np.random.uniform(0.0, 0.5, bins['0-50']))
    
    y_scores.extend(confidences)
    confidence_levels.extend(confidences)

# Convert to numpy arrays
y_true = np.array(y_true)
y_scores = np.array(y_scores)

# Binarize the output for multiclass ROC
y_true_bin = label_binarize(y_true, classes=range(NUM_CLASSES))
y_scores_bin = np.zeros_like(y_true_bin)

# Fill confidence scores
for i in range(NUM_CLASSES):
    class_mask = (y_true == i)
    y_scores_bin[class_mask, i] = y_scores[class_mask]

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(NUM_CLASSES):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_scores_bin[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_scores_bin.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Plot ROC curves
fig = go.Figure()
fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], 
                        mode='lines', 
                        name='Random',
                        line=dict(dash='dash')))

# Add micro-average
fig.add_trace(go.Scatter(
    x=fpr["micro"],
    y=tpr["micro"],
    mode='lines',
    name=f"Micro-average (AUC = {roc_auc['micro']:.2f})",
    line=dict(color='deeppink', width=4, dash='dot')
))

# Add macro-average
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(NUM_CLASSES)]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(NUM_CLASSES):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
mean_tpr /= NUM_CLASSES
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

fig.add_trace(go.Scatter(
    x=fpr["macro"],
    y=tpr["macro"],
    mode='lines',
    name=f"Macro-average (AUC = {roc_auc['macro']:.2f})",
    line=dict(color='navy', width=4, dash='dot')
))

# Add class-specific curves (limit to 5 for clarity)
colors = ["aqua", "darkorange", "cornflowerblue", "green", "red"]
for i, color in zip(range(min(5, NUM_CLASSES)), colors):
    fig.add_trace(go.Scatter(
        x=fpr[i],
        y=tpr[i],
        mode='lines',
        name=f"{CLASS_NAMES[i]} (AUC = {roc_auc[i]:.2f})",
        line=dict(color=color, width=2)
    ))

fig.update_layout(
    title='Multiclass ROC Curves (Simulated for Perfect Classifier)',
    xaxis_title='False Positive Rate',
    yaxis_title='True Positive Rate',
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
)

roc_plot_path = os.path.join(OUTPUT_DIR, f"roc_curve.{PLOT_FORMAT}")
fig.write_html(roc_plot_path)
print(f"ROC curve saved to: {roc_plot_path}")

# Save metrics to JSON
metrics = {
    "roc_auc": {
        "micro": roc_auc["micro"],
        "macro": roc_auc["macro"],
        "per_class": {CLASS_NAMES[i]: roc_auc[i] for i in range(NUM_CLASSES)}
    },
    "note": "ROC curves simulated based on confidence distribution since classifier achieved 100% accuracy",
    "class_names": CLASS_NAMES,
    "plots": {
        "roc_curve": roc_plot_path,
    }
}

with open(os.path.join(OUTPUT_DIR, "classification_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=4)

print("Evaluation metrics saved to classification_metrics.json")
print("\nSummary Metrics:")
print(f"Micro-average ROC AUC: {metrics['roc_auc']['micro']:.3f}")
print(f"Macro-average ROC AUC: {metrics['roc_auc']['macro']:.3f}")