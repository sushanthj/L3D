import numpy as np
import matplotlib.pyplot as plt

def dolly_zoom(fov_values, initial_distance):
    frames = len(fov_values)
    distance_values = np.zeros(frames)
    distance_values[0] = initial_distance

    for i in range(1, frames):
        # Calculate the new distance to maintain subject size constant
        distance_values[i] = distance_values[i-1] * np.tan(np.radians(fov_values[i-1] / 2)) / np.tan(np.radians(fov_values[i] / 2))

    return distance_values

# Define FOV range and initial distance
fov = np.linspace(5, 120, 10)
initial_distance = 50.0  # You can adjust the initial distance as needed

# Calculate dolly zoom effect
distance_values = dolly_zoom(fov, initial_distance)

# Plot the results
plt.plot(fov[1:], distance_values[1:], marker='o')
plt.title('Dolly Zoom Effect')
plt.xlabel('Field of View (FOV)')
plt.ylabel('Camera Distance')
plt.show()