import cv2
import numpy as np
import pyzed.sl as sl
import argparse

# DL library imports
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms
import torchvision.transforms.functional as TF
from collections import namedtuple

# import segmentation model
from seg_model.pspnet import PSPNet
import matplotlib.pyplot as plt


Label = namedtuple( "Label", [ "name", "train_id", "color"])
drivables = [ 
             Label("direct", 0, (0, 255, 0)),        # green
             Label("alternative", 1, (0, 0, 255)),  # blue
             Label("background", 2, (0, 0, 0)),        # black          
            ]
train_id_to_color = [c.color for c in drivables if (c.train_id != -1 and c.train_id != 255)]
train_id_to_color = np.array(train_id_to_color)
# Pixel to meter conversion factor
pixel_to_m_conversion_factor = 0.006818

def convert_pixels_to_m(pixels, conversion_factor):
    return pixels * conversion_factor

def segmentation(img):
    image_test = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    frame = cv2.resize(image_test, (640, 360))

    pt_image = preprocess(frame)
    pt_image = pt_image.to(device)

    # get model prediction and convert to corresponding color
    y_pred = torch.argmax(model(pt_image.unsqueeze(0)), dim=1).squeeze(0)
    predicted_labels = y_pred.cpu().detach().numpy()
    cm_labels = (train_id_to_color[predicted_labels]).astype(np.uint8)
    labels = cm_labels[:,:,1] 

    overlay_image = cv2.addWeighted(frame, 1, cm_labels, 0.25, 0)
    #overlay_image = cv2.cvtColor(overlay_image, cv2.COLOR_RGB2BGR)
    overlay_image = cv2.resize(overlay_image,(1000,500))
    
        
    ############## Calculating the fps
    # new_frame_time = time.time()
    # fps will be number of frame processed in given time frame
    # since their will be most of time error of 0.001 second
    # we will be subtracting it to get more accurate result
    # inference_time = (new_frame_time-prev_frame_time)
    # fps = 1/(new_frame_time-prev_frame_time)
    # prev_frame_time = new_frame_time
    
    # converting the fps into integer
    # fps = int(fps)

    # converting the fps to string so that we can display it on frame
    # by using putText function
    # fps = str(fps)
    # print("FPS: {} and inference time {:.2f}".format(fps, inference_time))

    # putting the FPS count on the frame
    # cv2.putText(overlay_image, "FPS="+fps, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 1, cv2.LINE_AA)

    ######### Show Output 
    # cv2.imshow("Drivespace_"+str(model_name), overlay_image)
    return overlay_image

def main():
    # Create a ZED camera object
    zed = sl.Camera()
    input_type = sl.InputType()
    if opt.svo is not None:
        input_type.set_from_svo_file(opt.svo)
    # Set the camera configuration
    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters(input_t=input_type, svo_real_time_mode=True)
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA  # QUALITY
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP_X_FWD
    init_params.depth_maximum_distance = 50

    # Open the camera
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit()
    print(f"status:{status}")
    # Create a runtime parameters object
    runtime_params = sl.RuntimeParameters()

    while True:
        # Capture a new frame from the camera
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            # Retrieve the left image from the camera
            image_zed = sl.Mat()
            zed.retrieve_image(image_zed, sl.VIEW.LEFT)

            # Convert the ZED image to OpenCV format
            raw_frame = image_zed.get_data()
            frame = segmentation(raw_frame)
            # Convert the frame to the HSV color space
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Define the lower and upper bounds for green color in HSV
            lower_green = np.array([50, 100, 100])
            upper_green = np.array([70, 255, 255])

            # Threshold the frame to extract the green region
            green_mask = cv2.inRange(hsv_frame, lower_green, upper_green)

            # Apply morphological operations to remove noise
            kernel = np.ones((5, 5), np.uint8)
            green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)

            # Find contours in the green mask
            green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # # Draw the detected green lanes on the original frame
            # for contour in green_contours:
            #     cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)

            # Calculate the width of the green region at each point
            if len(green_contours) > 0:
                # Get the largest green contour
                largest_contour = max(green_contours, key=cv2.contourArea)

                # Get the right and left edges of the largest green contour
                x_right = np.max(largest_contour[:, :, 0])
                x_left = np.min(largest_contour[:, :, 0])

                # Calculate the width from right to left edge in pixels
                green_width_pixels = x_right - x_left

                # Convert the width from pixels to meters
                green_width_m = convert_pixels_to_m(green_width_pixels, pixel_to_m_conversion_factor)

                # Draw lines to represent the width at each contour point
                for point in largest_contour:
                    x, y = point[0]
                    cv2.line(frame, (int(x), int(y)), (int(x), int(y + green_width_pixels)), (255, 0, 0), 2)

                # Print the width of the green region in meters
                print("Green region width:", green_width_m, "m")

            # Display the resulting frame
            cv2.imshow("Different Lane", frame)

            # Check for the 'q' key to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    zed.close()
    cv2.destroyAllWindows()


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--svo', type=str, default=None, help='optional svo file')
    opt = parser.parse_args()
    preprocess = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.485, 0.56, 0.406), std=(0.229, 0.224, 0.225))
                ])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    print(device)


    ###### CALL PSPNet
    model = PSPNet(in_channels=3, num_classes=3, use_aux=True).to(device)
    pretrained_weights = torch.load(f'PSPNet_res50_20.pt', map_location="cpu")
    model.load_state_dict(pretrained_weights)
    model_name= 'PSPNet'

    
    model.eval()
    print("model loaded")
    with torch.no_grad(): 
        main()

