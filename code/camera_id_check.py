import cv2

# Function to check if a camera is available
def is_camera_available(index):
    cap = cv2.VideoCapture(index)
    if cap.isOpened():
        cap.release()
        return True
    return False

# Get the IDs of all connected cameras
connected_cameras = []
index = 0
while True:
    try:
        if is_camera_available(index):
            connected_cameras.append(index)
        else:
            break
        index += 1
    except Exception as e:
        pass

# Print the list of connected camera IDs
print("Connected Camera IDs:", connected_cameras)
