import numpy as np
import matplotlib.pyplot as plt

# Number of epochs
epochs = 150

# Random seed for reproducibility
np.random.seed(42)

# Generate linear-exponential values with small jitters at every step
epoch_range = np.arange(epochs) + 1

# Function to create linear-exponential values with small jitters
def generate_linear_exponential_values(avg_final_value, jitter_magnitude):
    values = []
    for epoch in epoch_range:
        exponential_value = avg_final_value * (1 - np.exp(-epoch / 50))
        jitter = np.random.normal(0, jitter_magnitude, 1)[0]
        value = exponential_value + jitter
        if value > avg_final_value:
            value = avg_final_value
        values.append(value)
    return np.array(values)

# Small jitter magnitude
jitter_magnitude = 0.005

# Generate precision values with linear-exponential shape and small jitters
precision = generate_linear_exponential_values(0.945, jitter_magnitude)

# Generate recall values with linear-exponential shape and small jitters
recall = generate_linear_exponential_values(0.945, jitter_magnitude)

# Generate mAP values with linear-exponential shape and small jitters
mAP = generate_linear_exponential_values(0.945, jitter_magnitude)

# Generate F1 score values with linear-exponential shape and small jitters
f1 = generate_linear_exponential_values(0.945, jitter_magnitude)

# Create subplots
plt.figure(figsize=(6, 6))

# Plot the metrics
plt.subplot(2, 2, 1)
plt.plot(epoch_range, precision, label='Average Precision', color='blue')
plt.xlabel('Epochs')
plt.ylabel('Precision')
plt.title('Average Precision')

plt.subplot(2, 2, 2)
plt.plot(epoch_range, recall, label='Average Recall', color='green')
plt.xlabel('Epochs')
plt.ylabel('Recall')
plt.title('Average Recall')

plt.subplot(2, 2, 3)
plt.plot(epoch_range, mAP, label='Average mAP', color='red')
plt.xlabel('Epochs')
plt.ylabel('mAP')
plt.title('Average mAP')

plt.subplot(2, 2, 4)
plt.plot(epoch_range, f1, label='Average F1 Score', color='orange')
plt.xlabel('Epochs')
plt.ylabel('F1 Score')
plt.title('Average F1 Score')

# Summarized dataset information as a title for the entire figure
dataset_summary = "Dataset: XYZ (1000 images, 400 labels)"
plt.suptitle(dataset_summary, fontsize=10, color='gray')

plt.tight_layout()
plt.show()
