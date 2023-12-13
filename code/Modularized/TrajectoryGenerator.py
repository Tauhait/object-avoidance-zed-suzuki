import util
import CollisionAvoidance
import constant as const
import sys
import numpy as np
import time
import argparse
import torch
import cv2
import pyzed.sl as sl
import warnings, random
import math
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import ogl_viewer.viewer as gl
import cv_viewer.tracking_viewer as cv_viewer
import rospy
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from std_msgs.msg import Float32, String, Int8, Int32
from ultralytics import YOLO
from time import sleep
from math import asin, atan2, cos, degrees, radians, sin, sqrt
from torchvision import transforms
from collections import namedtuple
from novatel_oem7_msgs.msg import BESTPOS, BESTVEL, INSPVA
from seg_model.pspnet import PSPNet
from scipy.interpolate import CubicSpline
from threading import Lock, Thread
warnings.filterwarnings("ignore")


class TrajectoryGenerator:

    def __init__(self):
        # Initialize lane state variable
        self.lane_state = const.DRIVING_LANE

        # Initialize ROS subscriber and publisher for lane state
        rospy.Subscriber("/lane_state", Int8, self.callback_lane_state)
        self.lane_state_publish = rospy.Publisher('lane_state', Int32, queue_size=1)

        # Initialize device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load PSPNet model
        self.PSPNet_model = PSPNet(in_channels=3, num_classes=3, use_aux=True).to(self.device)
        pretrained_weights = torch.load("PSPNet_res50_20.pt", map_location="cpu")
        self.PSPNet_model.load_state_dict(pretrained_weights)
        self.PSPNet_model.eval()

        # Publish initial driving lane state
        self.lane_state_publish.publish(const.DRIVING_LANE)

        # Preprocess function
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.56, 0.406), std=(0.229, 0.224, 0.225))
        ])

        # Start the main program loop
        self.main()

    def callback_lane_state(self, data):
        """
        Callback function for lane state topic.
        """
        self.lane_state = data.data

    def randomise(self) -> float:
        """
        Generates a random value between 2.5 and 3.5.
        """
        random_value = random.uniform(2.5, 3.5)
        return random_value

    def xywh2abcd(self, xywh: np.ndarray, im_shape: tuple) -> np.ndarray:
        """
        Converts xywh format to abcd format.

        Args:
            xywh: BBox coordinates in xywh format (center x, center y, width, height).
            im_shape: Image shape (height, width).

        Returns:
            BBox coordinates in abcd format (top-left x, top-left y, bottom-right x, bottom-right y).
        """
        output = np.zeros((4, 2))

        # Center / Width / Height -> BBox corners coordinates
        x_min = (xywh[0] - 0.5*xywh[2]) #* im_shape[1]
        x_max = (xywh[0] + 0.5*xywh[2]) #* im_shape[1]
        y_min = (xywh[1] - 0.5*xywh[3]) #* im_shape[0]
        y_max = (xywh[1] + 0.5*xywh[3]) #* im_shape[0]

        # A ------ B
        # | Object |
        # D ------ C

        output[0][0] = x_min
        output[0][1] = y_min

        output[1][0] = x_max
        output[1][1] = y_min

        output[2][0] = x_min
        output[2][1] = y_max

        output[3][0] = x_max
        output[3][1] = y_max
        return output

    def get_class_label(self, det: np.ndarray) -> int:
        """
        Extracts and returns the class label from a detection array.

        Args:
            det: A NumPy array containing detection data.

        Returns:
            The class label as an integer.
        """
        # Assuming 'det' is a NumPy array, and you want to extract the first element
        number_str = str(det[0])

        # Now you can strip characters from the 'number_str'
        number_str = number_str.strip('[]').strip()

        # Convert the number string to an integer or use it as needed
        number = int(float(number_str))

        return number

    def get_angle_between_horizontal_base_object_center(self, 
        x1: float, x2: float, x3: float, x4: float, 
        y1: float, y2: float, y3: float, y4: float) -> float:
        """
        Calculates the angle between the horizontal base of an object and the positive x-axis.

        Args:
            x1, x2, x3, x4: Coordinates of the top-left, top-right, bottom-left, and bottom-right corners of the object.
            y1, y2, y3, y4: Corresponding y-coordinates of the corners.

        Returns:
            The angle in degrees.
        """
        vec1 = (x2-x1, y2-y1)
        vec2 = (x4-x3, y4-y3)

        dot_prod = vec1[0] * vec2[0] + vec1[1] * vec2[1]
        magnitude_vec1 = math.sqrt(vec1[0]**2 + vec1[1]**2)
        magnitude_vec2 = math.sqrt(vec2[0]**2 + vec2[1]**2)

        cos_angle = dot_prod / (magnitude_vec1*magnitude_vec2)
        angle_rad = math.acos(cos_angle)
        angle_deg = math.degrees(angle_rad)
        if angle_deg > 180:
            angle_deg -= 360
        elif angle_deg < -180:
            angle_deg += 360
        return angle_deg

    def get_green_masked_image(self, cm_labels: np.ndarray) -> tuple:
        """
        Masks green pixels in a semantic segmentation image.

        Args:
            cm_labels: Semantic segmentation image labels.

        Returns:
            A tuple containing a masked image with only green pixels and a green mask.
        """
        green_mask = (cm_labels[:,:,1] > 0) & (cm_labels[:,:,0] == 0) & (cm_labels[:,:,2] == 0)
        
        green_masked_image = np.zeros_like(cm_labels)
        green_masked_image[green_mask] = cm_labels[green_mask]
        return green_masked_image, green_mask

    def get_left_red_pixels(self, cm_labels: np.ndarray) -> tuple:
        """
        Masks red pixels in the left half of a semantic segmentation image.

        Args:
            cm_labels: Semantic segmentation image labels.

        Returns:
            A tuple containing a masked image with only red pixels in the left half and a red mask for the left half.
        """
        height, width, _ = cm_labels.shape
        red_mask = (cm_labels[:,:,2] > 0) & (cm_labels[:,:,1] == 0) & (cm_labels[:,:,0] == 0)
        # Create a mask for the right side of the matrix
        right_mask = np.zeros((height, width // 2), dtype=bool)
        # Combine the right mask and the red mask for the left half
        combined_red_mask = np.hstack((red_mask[:, :width // 2], right_mask))
        left_red_masked_image = np.zeros_like(cm_labels)
        left_red_masked_image[combined_red_mask] = cm_labels[combined_red_mask]
        return left_red_masked_image, combined_red_mask

    def get_right_red_pixels(self, cm_labels: np.ndarray) -> tuple:
        """
        Masks red pixels in the right half of a semantic segmentation image.

        Args:
            cm_labels: Semantic segmentation image labels.

        Returns:
            A tuple containing a masked image with only red pixels in the right half and a red mask for the right half.
        """
        height, width, _ = cm_labels.shape
        red_mask = (cm_labels[:,:,2] > 0) & (cm_labels[:,:,1] == 0) & (cm_labels[:,:,0] == 0)
        # Create a mask for the left side of the matrix
        left_mask = np.zeros((height, width // 2), dtype=bool)
        # Combine the left mask and the red mask for the right half
        combined_red_mask = np.hstack((left_mask, red_mask[:, width // 2:]))   
        red_masked_image = np.zeros_like(cm_labels)
        red_masked_image[combined_red_mask] = cm_labels[combined_red_mask]
        return red_masked_image, combined_red_mask

    def get_point_at_distance(self, lat: float, lon: float, d: float, bearing: float) -> tuple:
        """
        Calculates the latitude and longitude of a point at a given distance and bearing from another point.

        Args:
            lat: Initial latitude.
            lon: Initial longitude.
            d: Distance in kilometers.
            bearing: Heading in degrees.

        Returns:
            A tuple containing the final latitude and longitude.
        """
        d = d / 1000
        lat1 = radians(lat)
        lon1 = radians(lon)
        a = radians(bearing)
        lat2 = asin(sin(lat1) * cos(d/R) + cos(lat1) * sin(d/R) * cos(a))
        lon2 = lon1 + atan2(
            sin(a) * sin(d/R) * cos(lat1),
            cos(d/R) - sin(lat1) * sin(lat2)
        )
        return degrees(lat2), degrees(lon2)

    def get_next_overtake_waypoint(self, lat: float, lon: float) -> tuple:
        """
        Calculates the next waypoint for overtaking at a fixed distance and bearing.

        Args:
            lat: Initial latitude.
            lon: Initial longitude.

        Returns:
            A tuple containing the latitude and longitude of the next waypoint.
        """
        return self.get_point_at_distance(lat, lon, const.OVERTAKE_WAYPOINT_DIST, const.BEARING_ZERO)

    def is_clear_to_switch(self, overtake_lane_space: int) -> bool:
        """
        Checks if there is enough space in the overtake lane to switch.

        Args:
            overtake_lane_space: Fraction of the overtake lane occupied.

        Returns:
            True if there is enough space, False otherwise.
        """
        return overtake_lane_space > 2 * const.LANE_SPACE

    def get_object_depth_val(self, x: int, y: int, depth_map: sl.Camera) -> float:
        """
        Retrieves the depth value of an object at a specific pixel location in the depth map.

        Args:
            x: X-coordinate of the pixel.
            y: Y-coordinate of the pixel.
            depth_map: SLAM camera sensor object.

        Returns:
            The depth value at the specified pixel location in meters, or None if the depth is invalid.
        """
        _, center_depth = depth_map.get_value(x, y, sl.MEM.CPU)
        if center_depth not in [np.nan, np.inf, -np.inf]:
            # print(f"Depth value at center: {center_depth} metres.")  
            return center_depth
        return None
    
    def gen_trajectory(self, green_masked_image, red_masked_image, masked_image, depth_map) -> tuple:
        """
        Generates a trajectory for overtaking.

        Args:
            green_masked_image: Image with only green pixels masked.
            red_masked_image: Image with only red pixels masked.
            masked_image: Original image with combined masks.
            depth_map: SLAM camera sensor object.

        Returns:
            A tuple containing:
                - True if a valid trajectory is generated, False otherwise.
                - Updated masked image with the trajectory overlaid.
                - Distance to the free lane midpoint in meters.
        """

        try:
            # Extract midpoints of green and red areas
            green_midpoint = np.mean(np.where(green_masked_image), axis=1)
            red_midpoint = np.mean(np.where(red_masked_image), axis=1)

            # Check if midpoints are valid
            if all(not np.isnan(point) for point in red_midpoint) and all(not np.isnan(point) for point in green_midpoint):
                # Convert midpoints to integers
                red_midpoint_x = int(red_midpoint[1])
                red_midpoint_y = int(red_midpoint[0])

                # Calculate the maximum y-coordinate
                max_y = masked_image.shape[0] - 1

                # Convert green midpoint to integers
                green_midpoint_x = int(green_midpoint[1])
                green_midpoint_y = int(green_midpoint[0])

                # Calculate intermediate points for the trajectory
                green_red_midpoint_x = int((red_midpoint_x - green_midpoint_x) / 2)
                green_red_midpoint_y = int((max_y - red_midpoint_y) / 2)

                # Create arrays for the trajectory points
                x_array = np.array([green_midpoint_x, green_midpoint_x + green_red_midpoint_x, red_midpoint_x])
                y_array = np.array([max_y, red_midpoint_y + green_red_midpoint_y, red_midpoint_y])

                # Generate cubic spline object
                cubic_spline = CubicSpline(x_array, y_array)

                # Generate interpolated points for the trajectory
                interpolated_x = np.linspace(x_array[0], x_array[-1], const.NUM_INTERPOLATED_POINTS)
                interpolated_y = cubic_spline(interpolated_x)

                # Calculate distance to the free lane midpoint
                dist_to_free_lane_mid = self.get_object_depth_val(red_midpoint_x, red_midpoint_y, depth_map)

                # Handle invalid or exceeding distance
                if dist_to_free_lane_mid is None or np.isnan(dist_to_free_lane_mid) or dist_to_free_lane_mid > 5:
                    dist_to_free_lane_mid = self.randomise()

                # Overlay the trajectory on the masked image
                for x, y in zip(interpolated_x, interpolated_y):
                    x = int(x)
                    y = int(y)
                    if 0 <= y < masked_image.shape[0] and 0 <= x < masked_image.shape[1]:
                        masked_image[y, x] = [255, 255, 255]

                # Return success with updated image and distance
                return True, masked_image, dist_to_free_lane_mid
            else:
                print("Invalid midpoints detected, skipping trajectory generation.")
                return False, None, None
        except Exception as e:
            print(f"Exception occured: {e}, {type(e).__name__}")
            print("Returning False from generate_trajectory.")
            return False, None, None

    def is_clear_to_overtake(self, driving_lane_space, overtake_lane_space, green_masked_image, red_masked_image, masked_image, depth_map):
        """
        Checks if it is clear to overtake based on lane space and generates a trajectory if possible.

        Args:
            driving_lane_space: Fraction of the driving lane occupied.
            overtake_lane_space: Fraction of the overtake lane occupied.
            green_masked_image: Image with only green pixels masked.
            red_masked_image: Image with only red pixels masked.
            masked_image: Original image with combined masks.
            depth_map: SLAM camera sensor object.

        Returns:
            A tuple containing:
                - True if it is clear to overtake, False otherwise.
                - Updated masked image with the trajectory overlaid (if applicable).
                - Distance to the free lane midpoint in meters (if applicable).
        """

        if driving_lane_space >= 0 and driving_lane_space < const.LANE_SPACE and overtake_lane_space > const.LANE_SPACE:
            status, masked_image, dist_to_free_lane_mid = self.gen_trajectory(green_masked_image, red_masked_image, masked_image, depth_map)
            return status, masked_image, dist_to_free_lane_mid
        else:
            print("Lane space conditions not met for overtaking.")
            return False, None, None
        
    def drivespace(self, depth_map):
        """
        Analyzes the semantic segmentation image and determines the next action.

        Args:
            depth_map: SLAM camera sensor object.

        Returns:
            A masked image highlighting the driving space and any relevant information for overtaking.
        """
        freespace_frame = cv2.resize(cv2.cvtColor(image_net, cv2.COLOR_RGBA2RGB), (700, 500))

        pt_image = preprocess(freespace_frame)
        pt_image = pt_image.to(device)

        # get model prediction and convert to corresponding color
        y_pred = torch.argmax(PSPNet_model(pt_image.unsqueeze(0)), dim=1).squeeze(0)
        predicted_labels = y_pred.cpu().detach().numpy()

        cm_labels = (train_id_to_color[predicted_labels]).astype(np.uint8)
        
        green_masked_image, green_mask = util.get_green_masked_image(cm_labels)

        total_pixels = cm_labels.shape[0] * cm_labels.shape[1]
        num_green_pixels = np.sum(green_masked_image)
        normalized_num_green_pixels = num_green_pixels / total_pixels
        masked_image = np.zeros_like(cm_labels)
        driving_lane_space = int(normalized_num_green_pixels)
        
        if lane_state == const.DRIVING_LANE:
            red_masked_image, combined_red_mask = util.get_right_red_pixels(cm_labels)
            num_red_pixels = np.sum(red_masked_image)
            normalized_num_red_pixels = num_red_pixels / total_pixels
            combined_mask = green_mask | combined_red_mask
            overtake_lane_space = int(normalized_num_red_pixels)
            masked_image[combined_mask] = cm_labels[combined_mask]

            status, updated_mask, dist_to_free_lane_mid = util.is_clear_to_overtake(driving_lane_space, overtake_lane_space, green_masked_image, red_masked_image, masked_image, depth_map)
            if status == False or dist_to_free_lane_mid == None or np.isnan(dist_to_free_lane_mid):
                collision_avoidance_publish.publish(const.CONTINUE)
                overtake_lat_lon_publish.publish('0,0,0')
            else:
                print(f"dist_to_free_lane_mid: {dist_to_free_lane_mid}")
                masked_image = updated_mask
                overtake_bearing = util.get_bearing(dist_to_free_lane_mid)
                overtake_x, overtake_y = util.get_point_at_distance(lat, lon, dist_to_free_lane_mid, overtake_bearing, R=6371)
                collision_avoidance_publish.publish(const.OVERTAKE)
                overtake_lat_lon_publish.publish(f'{str(dist_to_free_lane_mid)},{str(overtake_x)},{str(overtake_y)}')
        elif lane_state == const.OVERTAKE_LANE:
            red_masked_image, combined_red_mask = util.get_left_red_pixels(cm_labels)
            num_red_pixels = np.sum(red_masked_image)
            normalized_num_red_pixels = num_red_pixels / total_pixels
            combined_mask = green_mask | combined_red_mask
            overtake_lane_space = int(normalized_num_red_pixels)
            masked_image[combined_mask] = cm_labels[combined_mask]
            if util.is_clear_to_switch(overtake_lane_space):
                collision_avoidance_publish.publish(const.CONTINUE)
                lane_state_publish.publish(const.DRIVING_LANE)

        
        return masked_image

class TrajectoryGenerator:


    def gen_trajectory(self, green_masked_image, red_masked_image, masked_image, depth_map):
        try:
            green_midpoint = np.mean(np.where(green_masked_image), axis=1)
            red_midpoint = np.mean(np.where(red_masked_image), axis=1)
            if(not np.isnan(red_midpoint[0]) and not np.isnan(red_midpoint[1]) and not np.isnan(green_midpoint[1]) and not np.isnan(green_midpoint[0])):
                red_midpoint_x = int(red_midpoint[1])  
                red_midpoint_y = int(red_midpoint[0]) 
                # dist_to_free_lane_mid = get_object_depth_val(red_midpoint_x, red_midpoint_y, depth_map)
                max_y = masked_image.shape[0] - 1
                green_midpoint_x = int(green_midpoint[1])
                green_midpoint_y = int(green_midpoint[0])
                green_red_midpoint_x = int((red_midpoint_x - green_midpoint_x) / 2)
                green_red_midpoint_y = int((max_y - red_midpoint_y)/ 2)
            
                x_array = np.array([green_midpoint_x, green_midpoint_x + green_red_midpoint_x, red_midpoint_x])
                y_array = np.array([max_y, red_midpoint_y + green_red_midpoint_y, red_midpoint_y])

                cubic_spline = CubicSpline(x_array, y_array)

                interpolated_x = np.linspace(x_array[0], x_array[-1], const.NUM_INTERPOLATED_POINTS)
                interpolated_y = cubic_spline(interpolated_x)
                dist_to_free_lane_mid = self.get_object_depth_val(red_midpoint_x, red_midpoint_y, depth_map)
                if dist_to_free_lane_mid == None or np.isnan(dist_to_free_lane_mid) or dist_to_free_lane_mid > 5:
                    dist_to_free_lane_mid = self.randomise()
                # Set pixels along the trajectory to white
                for x, y in zip(interpolated_x, interpolated_y):
                    x = int(x)
                    y = int(y)
                    if 0 <= y < masked_image.shape[0] and 0 <= x < masked_image.shape[1]:
                        masked_image[y, x] = [255, 255, 255]
                return True, masked_image, dist_to_free_lane_mid
        except Exception as e:
            print(f"Exception occured : {e}, {type(e).__name__}")
        print(f"Return False from is_clear_to_overtake")
        return False, None, None

    def is_clear_to_overtake(self, driving_lane_space, overtake_lane_space, green_masked_image, red_masked_image, masked_image, depth_map):
        if driving_lane_space >= 0 and driving_lane_space < const.LANE_SPACE and overtake_lane_space > const.LANE_SPACE:
            status, masked_image, dist_to_free_lane_mid = self.gen_trajectory(green_masked_image, red_masked_image, masked_image, depth_map)
            return status, masked_image, dist_to_free_lane_mid    
        return False, None, None