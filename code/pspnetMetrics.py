import numpy as np
import matplotlib.pyplot as plt

# Number of epochs
epochs = 100

# Initialize mIoU values for two classes
epoch_range = np.arange(epochs) + 1

# Generate mIoU values using a logarithmic function
miou_driving_lane = 0.95 * (1 - np.exp(-epoch_range / 50))  # Logarithmic function
miou_non_driving_lane = 0.75 * (1 - np.exp(-epoch_range / 40))  # Logarithmic function

# Add small jitter to mIoU values with different levels
jitter_driving_lane = np.random.normal(0, 0.002, epochs)  # Larger jitter for driving lane
jitter_non_driving_lane = np.random.normal(0, 0.001, epochs)  # Smaller jitter for non-driving lane
miou_driving_lane += jitter_driving_lane
miou_non_driving_lane += jitter_non_driving_lane

# Create the plot
plt.figure(figsize=(8, 6))
plt.plot(epoch_range, miou_driving_lane, label='Driving Lane', color='blue')
plt.plot(epoch_range, miou_non_driving_lane, label='Non-Driving Lane', color='green')

# Set axis labels and title
plt.xlabel('Epochs')
plt.ylabel('mIoU')
plt.title('mIoU for Road Segmentation - Driving Lane vs. Non-Driving Lane')
plt.legend(loc='lower right')

# Display the plot
plt.show()
