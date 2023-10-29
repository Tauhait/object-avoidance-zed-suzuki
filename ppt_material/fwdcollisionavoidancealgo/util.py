import random, numpy as np, math
import constant as const
import pyzed.sl as sl
from scipy.interpolate import CubicSpline
from math import asin, atan2, cos, degrees, radians, sin, sqrt

def get_speed(collision_warning, lane_state, bearing_diff):
    # print("Inside get_speed")
    const_speed = const.DRIVE_SPEED
    if lane_state == const.DRIVING_LANE:
        if abs(bearing_diff) > const.TURN_BEARNG_THRESHOLD:
            print(f"bearing diff > turn bearing threshold")
            if(collision_warning == 2):
                const_speed = const.BRAKE_SPEED
            else:
                const_speed = const.OVERTAKE_SPEED
        elif collision_warning == 2:
            const_speed = const.BRAKE_SPEED
        else:
            const_speed = const.DRIVE_SPEED
    elif lane_state == const.CHANGE_LANE:
        if collision_warning == 2:
            const_speed = const.BRAKE_SPEED
        else:
            const_speed = const.CHANGE_SPEED
    else:
        if collision_warning == 2:
            const_speed = const.BRAKE_SPEED
        else:
            const_speed = const.OVERTAKE_SPEED

    return const_speed
            

    
    
def randomise():
    random_value = random.uniform(2.5, 3.5)
    return random_value

def xywh2abcd(xywh, im_shape):
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

def get_class_label(det):
    # Assuming 'det' is a NumPy array, and you want to extract the first element
    number_str = str(det[0])

    # Now you can strip characters from the 'number_str'
    number_str = number_str.strip('[]').strip()

    # Convert the number string to an integer or use it as needed
    number = int(float(number_str))

    return number

def get_angle_between_horizontal_base_object_center(x1, x2, x3, x4, y1, y2, y3, y4):
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

def get_green_masked_image(cm_labels):
    green_mask = (cm_labels[:,:,1] > 0) & (cm_labels[:,:,0] == 0) & (cm_labels[:,:,2] == 0)
    
    green_masked_image = np.zeros_like(cm_labels)
    green_masked_image[green_mask] = cm_labels[green_mask]
    return green_masked_image, green_mask

def get_left_red_pixels(cm_labels):
    height, width, _ = cm_labels.shape
    red_mask = (cm_labels[:,:,2] > 0) & (cm_labels[:,:,1] == 0) & (cm_labels[:,:,0] == 0)
    # Create a mask for the right side of the matrix
    right_mask = np.zeros((height, width // 2), dtype=bool)
    # Combine the right mask and the red mask for the left half
    combined_red_mask = np.hstack((red_mask[:, :width // 2], right_mask))
    left_red_masked_image = np.zeros_like(cm_labels)
    left_red_masked_image[combined_red_mask] = cm_labels[combined_red_mask]
    return left_red_masked_image, combined_red_mask

def get_right_red_pixels(cm_labels):
    height, width, _ = cm_labels.shape
    red_mask = (cm_labels[:,:,2] > 0) & (cm_labels[:,:,1] == 0) & (cm_labels[:,:,0] == 0)
    # Create a mask for the left side of the matrix
    left_mask = np.zeros((height, width // 2), dtype=bool)
    # Combine the left mask and the red mask for the right half
    combined_red_mask = np.hstack((left_mask, red_mask[:, width // 2:]))   
    red_masked_image = np.zeros_like(cm_labels)
    red_masked_image[combined_red_mask] = cm_labels[combined_red_mask]
    return red_masked_image, combined_red_mask


def get_bearing(hyp, base=1.6):
    bearing = 0
    if hyp >= base:
        perp = math.sqrt(hyp**2 - base**2)
        bearing = math.atan(perp / base)
    return 90 - math.degrees(bearing)

def get_point_at_distance(lat, lon, d, bearing, R=6371):
    """
    lat: initial latitude, in degrees
    lon: initial longitude, in degrees
    d: target distance from initial
    bearing: (true) heading in degrees
    R: optional radius of sphere, defaults to mean radius of earth

    Returns new lat/lon coordinate {d}km from initial, in degrees
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

def get_object_depth_val(x, y, depth_map):
    _, center_depth = depth_map.get_value(x, y, sl.MEM.CPU)
    if center_depth not in [np.nan, np.inf, -np.inf]:
        # print(f"Depth value at center: {center_depth} metres.")  
        return center_depth
    return None

def gen_trajectory(green_masked_image, red_masked_image, masked_image, depth_map):
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
            dist_to_free_lane_mid = get_object_depth_val(red_midpoint_x, red_midpoint_y, depth_map)
            if dist_to_free_lane_mid == None or np.isnan(dist_to_free_lane_mid) or dist_to_free_lane_mid > 5:
                dist_to_free_lane_mid = randomise()
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

def is_clear_to_overtake(driving_lane_space, overtake_lane_space, green_masked_image, red_masked_image, masked_image, depth_map):
    if driving_lane_space >= 0 and driving_lane_space < const.LANE_SPACE and overtake_lane_space > const.LANE_SPACE:
        status, masked_image, dist_to_free_lane_mid = gen_trajectory(green_masked_image, red_masked_image, masked_image, depth_map)
        return status, masked_image, dist_to_free_lane_mid    
    return False, None, None

def set_angle(m_nAngle, deltaAngle):
    m_nAngle = m_nAngle + deltaAngle
    gearRatio = 17.75
    if (40 * gearRatio < m_nAngle):
        m_nAngle = 40 * gearRatio
    elif (-40 * gearRatio > m_nAngle):
        m_nAngle = -40 * gearRatio
    l_usAngle = (m_nAngle / gearRatio - (-65.536)) / 0.002

    H_Angle = (int)(l_usAngle) >> 8
    L_Angle = (int)(l_usAngle) & 0xff
    return H_Angle, L_Angle

def calc_checksum(msg):
    cs = 0
    for m in msg:
        cs += m
    cs = (0x00 - cs) & 0x000000FF
    cs = cs & 0xFF
    return cs

def set_speed(speed):
    speed = speed * 128
    H_speed = (int)(speed) >> 8
    L_speed = (int)(speed) & 0xff
    return H_speed, L_speed

def get_msg_to_mabx(speed, m_nAngle, angle, flasher, counter):
    H_Angle, L_Angle = set_angle(m_nAngle, -1*angle)
    H_Speed, L_Speed = set_speed(speed)

    msg_list = [1, counter, 0, 1, 52, 136, 215, 1, H_Speed, L_Speed, H_Angle, L_Angle, 0, flasher, 0, 0, 0, 0]

    msg_list[2] = calc_checksum(msg_list)
    message = bytearray(msg_list)
    print("Speed: ", message[8], message[9])
    print("Angle: ", message[10], message[11])
    print("===============================================================================")
    return message

def get_flasher(angle):
    return 1 if angle > 90 else 2 if angle < -100 else 0

def has_reached(current_loc, target_loc):
    distance_to_target = np.linalg.norm(np.array(current_loc) - target_loc) * const.LAT_LNG_TO_METER
    return distance_to_target < const.TARGET_REACH

def get_next_overtake_waypoint(lat, lon):
    return get_point_at_distance(lat, lon, const.OVERTAKE_WAYPOINT_DIST, const.BEARING_ZERO)

def is_clear_to_switch(overtake_lane_space):
    return overtake_lane_space > 2 * const.LANE_SPACE

def get_coordinates(file_path):
    coordinates_list = []
    try:
        with open(file_path, 'r') as file:
            for line in file:
                try:
                    coordinates = [float(coord) for coord in line.strip().strip('[],').split(',')]
                    coordinates_list.append(coordinates)
                except ValueError:
                    # Handle the exception if a value cannot be converted to float
                    print(f"Error: Unable to convert coordinates in line '{line}' to float.")
    except FileNotFoundError:
        # Handle the exception if the file is not found
        print(f"Error: The file '{file_path}' could not be found.")
    except Exception as e:
        # Handle any other unexpected exceptions
        print(f"An error occurred: {e}")
    
    return coordinates_list