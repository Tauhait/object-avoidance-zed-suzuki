import random, numpy as np, math
import os
import logging
import datetime
import constant as const
import pyzed.sl as sl
from scipy.interpolate import CubicSpline
from math import asin, atan2, cos, degrees, radians, sin, sqrt


def calculate_bearing_difference_for_speed_reduction(current_location, current_bearing, heading, WAYPOINTS, wp, WP_LEN):
    next_wp = 4                 # changed from 3 to 4
    
    if wp + next_wp < WP_LEN:
        off_y = - current_location[0] + WAYPOINTS[wp+next_wp][0]
        off_x = - current_location[1] + WAYPOINTS[wp+next_wp][1]
    else:
        off_y = - current_location[0] + WAYPOINTS[wp][0]
        off_x = - current_location[1] + WAYPOINTS[wp][1]
        
    # calculate bearing based on position error
    target_bearing = 90.00 + math.atan2(-off_y, off_x) * const.RAD_TO_DEG_CONVERSION

    # convert negative bearings to positive by adding 360 degrees
    if target_bearing < 0:
        target_bearing += 360.00
    
    current_bearing = heading 
    while current_bearing is None:
        current_bearing = heading 

    current_bearing = float(current_bearing)
    future_bearing_diff = current_bearing - target_bearing
    
    # normalize bearing difference to range between -180 and 180 degrees
    if future_bearing_diff < -180:
        future_bearing_diff += 360
    elif future_bearing_diff > 180:
        future_bearing_diff -= 360 

    # log_and_print(f"    Future Bearing Difference for ({next_wp}) waypoint : {future_bearing_diff:.1f}")
    return future_bearing_diff

def calculate_steer_output(current_location, current_bearing, heading, wp, WAYPOINTS):
    steer_gain = const.STEER_GAIN
    off_y = - current_location[0] + WAYPOINTS[wp][0]
    off_x = - current_location[1] + WAYPOINTS[wp][1]

    # calculate bearing based on position error
    target_bearing = 90.00 + math.atan2(-off_y, off_x) * const.RAD_TO_DEG_CONVERSION 

    # convert negative bearings to positive by adding 360 degrees
    if target_bearing < 0:
        target_bearing += 360.00
    
    current_bearing = heading 
    while current_bearing is None:
        current_bearing = heading 
    current_bearing = float(current_bearing)

    # calculate the difference between heading and bearing
    bearing_diff = current_bearing - target_bearing

    # normalize bearing difference to range between -180 and 180 degrees
    if bearing_diff < -180:
        bearing_diff += 360
    elif bearing_diff > 180:
        bearing_diff -= 360
    
    if abs(bearing_diff) < 1:           #Nullify the small the bearing difference
        bearing_diff = 0
        steer_gain = 300
        # log_and_print(f"    Nullied | Bearing Difference of {temp:.1f}")
    elif abs(bearing_diff) > 20:
        steer_gain = 1200
    
    steer_output = steer_gain * np.arctan(-1 * 2 * 3.5 * np.sin(np.radians(bearing_diff)) / 8)
    return steer_output, bearing_diff

def setup_logging(log_dir, file_path):
    # Get the base filename from the file path
    base_filename = os.path.basename(file_path)
    # Remove the file extension to get the desired string
    logFileName = os.path.splitext(base_filename)[0]

    # Create the log directory if it does not exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Set up the logging format
    log_format = "%(asctime)s [%(levelname)s]: %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=log_format)

    # Create a log file with the current timestamp as the name
    log_filename = logFileName + "-" + datetime.datetime.now().strftime("%d-%m-%Y :: %H,%M,%S") + ".log"
    log_path = os.path.join(log_dir, log_filename)

    # Add a file handler to save logs to the file
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)

    # Set the log format for the file handler
    file_handler.setFormatter(logging.Formatter(log_format))
    # file_handler.flush = True

    # Add the file handler to the logger
    logger = logging.getLogger('')
    logger.addHandler(file_handler)

    return logger

def calculate_steer_output_change_lane(currentLocation, targetLocation, current_bearing, heading):
    off_y = - currentLocation[0] + targetLocation[0]
    off_x = - currentLocation[1] + targetLocation[1]

    # calculate bearing based on position error
    target_bearing = 90.00 + math.atan2(-off_y, off_x) * const.RAD_TO_DEG_CONVERSION 

    # convert negative bearings to positive by adding 360 degrees
    if target_bearing < 0:
        target_bearing += 360.00
    
    current_bearing = heading 
    while current_bearing is None:
        current_bearing = heading 
    current_bearing = float(current_bearing)

    # calculate the difference between heading and bearing
    bearing_diff = current_bearing - target_bearing

    # normalize bearing difference to range between -180 and 180 degrees
    if bearing_diff < -180:
        bearing_diff = bearing_diff + 360

    if bearing_diff > 180:
        bearing_diff = bearing_diff - 360 

    steer_output = const.STEER_GAIN * np.arctan(-1 * 2 * 3.5 * np.sin(np.radians(bearing_diff)) / 8)
    
    return steer_output, bearing_diff

def get_speed(collision_warning, lane_state, bearing_diff):
    print("Inside get_speed(collision_warning, lane_state, bearing_diff) method")
    const_speed = const.DRIVE_SPEED
    if lane_state == const.DRIVING_LANE:
        print(f"Bearing Diff = {bearing_diff}")
        if abs(bearing_diff) > const.TURN_BEARNG_THRESHOLD:
            print(f"bearing diff > turn bearing threshold")
            if(collision_warning == const.URGENT_WARNING):
                const_speed = const.BRAKE_SPEED
            else:
                const_speed = const.OVERTAKE_SPEED
        elif collision_warning == const.URGENT_WARNING:
            const_speed = const.BRAKE_SPEED
        else:
            const_speed = const.DRIVE_SPEED
    elif lane_state == const.CHANGE_LANE:
        if collision_warning == const.URGENT_WARNING:
            const_speed = const.BRAKE_SPEED
        else:
            const_speed = const.CHANGE_SPEED 
        # const_speed = const.CHANGE_SPEED 
    else:
        if collision_warning == const.URGENT_WARNING:
            const_speed = const.BRAKE_SPEED
        else:
            const_speed = const.OVERTAKE_SPEED

    print(f"Inside get_speed(collision_warning, lane_state, bearing_diff) const_speed = {const_speed}")
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
    if hyp is None:
        return 0
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
    if d is None:
        return degrees(0), degrees(0)
    d = d / 1000
    lat1 = radians(lat)
    lon1 = radians(lon)
    a = radians(bearing)
    lat2 = asin(sin(lat1) * cos(d/R) + cos(lat1) * sin(d/R) * cos(a))
    lon2 = lon1 + atan2(
        sin(a) * sin(d/R) * cos(lat1),
        cos(d/R) - sin(lat1) * sin(lat2)
    )
    print(f"degrees(lat2) = {degrees(lat2)}, degrees(lon2) = {degrees(lon2)}")
    return degrees(lat2), degrees(lon2)

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
    return message

def get_flasher(angle):
    return 1 if angle > 90 else 2 if angle < -100 else 0

def has_reached(current_loc, target_loc):
    distance_to_target = np.linalg.norm(np.array(current_loc) - target_loc) * const.LAT_LNG_TO_METER
    return distance_to_target < const.TARGET_REACH

def get_next_overtake_waypoint(lat, lon):
    return get_point_at_distance(lat, lon, const.OVERTAKE_WAYPOINT_DIST, const.BEARING_ZERO)

def is_clear_to_switch(overtake_lane_space):
    return overtake_lane_space > 2 * const.OVERTAKE_LANE_SPACE

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
        print(f"An error occurred from get_coordinates() function: {e}")
    
    return coordinates_list

def get_point_at_distance(lat1, lon1, d, bearing, R=6371):
    """
    lat: initial latitude, in degrees
    lon: initial longitude, in degrees
    d: target distance from initial
    bearing: (true) heading in degrees
    R: optional radius of sphere, defaults to mean radius of earth

    Returns new lat/lon coordinate {d}km from initial, in degrees
    """
    lat1 = radians(lat1)
    lon1 = radians(lon1)
    a = radians(bearing)
    lat2 = asin(sin(lat1) * cos(d/R) + cos(lat1) * sin(d/R) * cos(a))
    lon2 = lon1 + atan2(
        sin(a) * sin(d/R) * cos(lat1),
        cos(d/R) - sin(lat1) * sin(lat2)
    )
    return (degrees(lat2), degrees(lon2))


def gen_trail_of_waypoints(obs_lat, obs_lon, ob_class):
    # Number of waypoints in the sequence
    # Check if the file exists
    if os.path.exists(const.FILENAME_TRAIL):
        # Delete the file
        os.remove(const.FILENAME_TRAIL)
    
    dist = []
    bear = []
    if ob_class == const.OBJ_CLASS_CAR:
        num_waypoints = len(const.DISTANCE_CAR)
        dist = const.DISTANCE_CAR
        bear = const.BEARING_CAR

    elif ob_class == const.OBJ_CLASS_CYCLE:
        num_waypoints = len(const.DISTANCE_CYCLE)
        dist = const.DISTANCE_CYCLE
        bear = const.BEARING_CYCLE
    else:
        num_waypoints = len(const.DISTANCE_PED)
        dist = const.DISTANCE_PED
        bear = const.BEARING_PED
    print(f"num_waypoints = {num_waypoints}")
    with open(const.FILENAME_TRAIL, 'w') as file:
        overtake_lat, overtake_lon = get_point_at_distance(obs_lat, obs_lon, dist[0], bear[0])
        file.write(f"[{overtake_lat}, {overtake_lon}]\n")
        
        for n in range(1, num_waypoints):
            print(f"{overtake_lat}, {overtake_lon}")
            gen_lat, gen_lon = get_point_at_distance(overtake_lat, overtake_lon, dist[n], bear[n])
            file.write(f"[{gen_lat}, {gen_lon}]\n")
            overtake_lat = gen_lat
            overtake_lon = gen_lon
    