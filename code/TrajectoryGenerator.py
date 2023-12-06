import random, numpy as np, math
import constant as const
import pyzed.sl as sl
from scipy.interpolate import CubicSpline
from math import asin, atan2, cos, degrees, radians, sin, sqrt

class TrajectoryGenerator:
    def randomise(self):
        random_value = random.uniform(2.5, 3.5)
        return random_value
    
    def get_object_depth_val(self, x, y, depth_map):
        _, center_depth = depth_map.get_value(x, y, sl.MEM.CPU)
        if center_depth not in [np.nan, np.inf, -np.inf]:
            # print(f"Depth value at center: {center_depth} metres.")  
            return center_depth
        return None

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