import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Generate synthetic data with varying levels of discrimination
np.random.seed(0)
n_samples = 100
y_true = np.random.randint(2, size=n_samples)
y_scores = np.random.rand(n_samples)

# Add some discriminatory power
y_scores[y_true == 1] += np.random.normal(loc=0.5, scale=0.2, size=len(y_scores[y_true == 1]))

# Compute ROC curve
fpr, tpr, _ = roc_curve(y_true, y_scores)

# Compute the area under the curve (AUC)
roc_auc = auc(fpr, tpr)

# Create the ROC curve plot
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.fill_between(fpr, 0, tpr, color='lightgray', alpha=0.5)  # Fill the area under the ROC curve
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Back Camera - Dense Optical Flow ROC Curve')
plt.legend(loc='lower right')
plt.show()
