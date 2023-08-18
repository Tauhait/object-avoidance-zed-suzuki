import csv
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.basemap import Basemap
import numpy as np

# Read data from CSV using Pandas
data = pd.read_csv('ca_test2.csv')

# Extract Latitude, Longitude, and Steering Angle
latitudes = data['Latitude'].tolist()
longitudes = data['Longitude'].tolist()
steering_angles = data['Steering_Angle'].tolist()

# Create a map-like plot
fig = plt.figure(figsize=(12, 8))
m = Basemap(projection='merc', llcrnrlat=min(latitudes), urcrnrlat=max(latitudes),
            llcrnrlon=min(longitudes), urcrnrlon=max(longitudes), resolution='i')
m.drawcoastlines()
m.drawcountries()

# Convert latitudes and longitudes to map coordinates
x, y = m(longitudes, latitudes)

# Plot data points on the map
m.scatter(x, y, marker='o', color='g', label='Data Points')
for i, angle in enumerate(steering_angles):
    plt.annotate(f"{angle:.2f}°", (x[i], 1+y[i]), fontsize=8, color='r')

plt.title('Latitude-Longitude Map with Steering Angles')
plt.legend(loc='lower right')

plt.show()

# Plot the steering angle data
plt.figure(figsize=(10, 6))
plt.plot(steering_angles, marker='o', linestyle='-', color='b')
plt.xlabel('Data Point')
plt.ylabel('Steering Angle')
plt.title('Steering Angle Data')
plt.grid(True)
plt.show()

# Calculate statistical measures
mean_angle = np.mean(steering_angles)
max_angle = np.max(steering_angles)
min_angle = np.min(steering_angles)
std_dev = np.std(steering_angles)
median_angle = np.median(steering_angles)

# Print the calculated statistics
print("Mean Angle:", mean_angle)
print("Max Angle:", max_angle)
print("Min Angle:", min_angle)
print("Standard Deviation:", std_dev)
print("Median Angle:", median_angle)