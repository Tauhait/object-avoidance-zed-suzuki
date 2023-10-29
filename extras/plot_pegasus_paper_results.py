import matplotlib.pyplot as plt

# Data
iterations = list(range(1, 11))
time_to_stop = [2.5, 3.0, 2.8, 2.3, 3.2, 2.6, 2.9, 2.7, 2.4, 3.1]
deceleration_rate = [4.0, 3.0, 3.5, 4.5, 2.8, 3.8, 3.3, 3.6, 4.2, 2.9]

# Create the figure and axis
fig, ax1 = plt.subplots(figsize=(8, 6))

# Plot "Time to stop" on the first y-axis
ax1.plot(iterations, time_to_stop, marker='o', linestyle='-', color='b', label='Time to Stop')
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Time to Stop (seconds)', color='b')
ax1.tick_params(axis='y', labelcolor='b')
ax1.grid(True)

# Create a second y-axis for "Deceleration Rate"
ax2 = ax1.twinx()
ax2.plot(iterations, deceleration_rate, marker='s', linestyle='--', color='r', label='Deceleration Rate')
ax2.set_ylabel('Deceleration Rate (m/s^2)', color='r')
ax2.tick_params(axis='y', labelcolor='r')

# Add legends
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc='upper left')

# Show the plot
plt.title('Time to Stop and Deceleration Rate vs. Iteration')
plt.show()