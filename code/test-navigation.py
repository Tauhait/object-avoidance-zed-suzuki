# Navigation in Development (SpeedBump with IMU also)
# Author : Rishav KUMAR (Mtech AI)

import math
import os
import logging
import datetime
import socket
import time
import novatel_oem7_msgs
import numpy as np
import rospy
from novatel_oem7_msgs.msg import BESTPOS, BESTVEL, INSPVA
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32, String

# Define the MABX IP address and port for sending data
mabx_IP = "192.168.50.1"
mabx_PORT = 30000

# Define buffer size and local interface
BUFFER_SIZE = 4096
local_interface = 'eth0'

# Conversion factor for latitude and longitude to meters
LAT_LNG_TO_METER = 1.111395e5
CW_flag = 0

# Initialize the ROS node for the algorithm
rospy.init_node('navigation', anonymous=True)

# Initialize the UDP socket for MABX communication
mabx_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
mabx_addr = (mabx_IP, mabx_PORT)

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

def log_and_print(str):
    logger.info(str)
    print(str)

# Function to parse and retrieve coordinates from a file
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

def callback_flag(data):
    global CW_flag
    CW_flag = 0
    CW_flag = data.data

def callback_pothole(data):
    global pothole_flag
    pothole_flag = 0
    pothole_flag = data.data
   
def callback_zed_camera_imu(data):
    global acc_z
    acc_z = 0 
    acc_z = data.linear_acceleration.z


rospy.Subscriber("/collision", Float32, callback_flag)   
rospy.Subscriber("/pothole", Float32, callback_pothole)
rospy.Subscriber("/imu/data_raw", Imu, callback_zed_camera_imu) 

# Callback functions for handling GNSS data
def callback_velocity(data):
    global current_vel
    current_vel = 3.6 * data.hor_speed 

def callback_heading(data):
    global heading
    heading = data.azimuth  #Left-handed rotation around z-axis in degrees clockwise from North. 

def callback_latlng(data):
    global lat, lng, lat_delta, lng_delta
    lat = data.lat
    lng = data.lon
    lat_delta = data.lat_stdev
    lng_delta = data.lon_stdev
   
rospy.Subscriber("/novatel/oem7/bestvel",BESTVEL, callback_velocity)    
rospy.Subscriber("/novatel/oem7/inspva",INSPVA, callback_heading) 
rospy.Subscriber("/novatel/oem7/bestpos",BESTPOS, callback_latlng)

time.sleep(0.1)

# Function to calculate and set steering angle based on current angle and target angle
def set_angle(current_angle, angle_change):
    # Steering gear ratio
    gear_ratio = 17.75
    # Limit steering angle within a range
    if (40 * gear_ratio < current_angle):
        current_angle = 40 * gear_ratio
    elif (-40 * gear_ratio > current_angle):
        current_angle = -40 * gear_ratio
    # Calculate scaled angle for transmission
    scaled_angle = (current_angle / gear_ratio - (-65.536)) / 0.002
    high_angle_byte, low_angle_byte = (int)(scaled_angle) >> 8, (int)(scaled_angle) & 0xff
    return high_angle_byte, low_angle_byte

def calc_checksum(message_bytes):
    checksum = 0
    for m in message_bytes:
        checksum += m
    checksum = (0x00 - checksum) & 0x000000FF
    checksum = checksum & 0xFF
    return checksum

def set_speed(vehicle_speed):
    vehicle_speed = vehicle_speed * 128
    high_byte_speed = (int)(vehicle_speed) >> 8
    low_byte_speed = (int)(vehicle_speed) & 0xff
    return high_byte_speed, low_byte_speed

def send_message_to_mabx(speed, current_angle, target_angle, flasher_light, message_counter):
    H_Angle, L_Angle = set_angle(current_angle, -1 * target_angle)
    H_Speed, L_Speed = set_speed(speed)

    message_bytes = [1, message_counter, 0, 1, 52, 136, 215, 1, H_Speed, L_Speed, H_Angle, L_Angle, 0, flasher_light, 0, 0, 0, 0]
    message_bytes[2] = calc_checksum(message_bytes)
    message = bytearray(message_bytes)
    #print("Reading the sent Speed from MABX: ", message[8], message[9])
    #print("Reading the sent Angle from MABX: ", message[10], message[11])
    log_and_print("================================================================")
    return message

def calculate_steer_output(currentLocation, Current_Bearing):
    global wp
    RAD_TO_DEG_CONVERSION = 57.2957795
    STEER_GAIN = 900         # For Tight Turns 1200 can be used

    off_y = - currentLocation[0] + waypoints[wp][0]
    off_x = - currentLocation[1] + waypoints[wp][1]

    # calculate bearing based on position error
    target_bearing = 90.00 + math.atan2(-off_y, off_x) * RAD_TO_DEG_CONVERSION

    # convert negative bearings to positive by adding 360 degrees
    if target_bearing < 0:
        target_bearing += 360.00
    
    Current_Bearing = heading 
    while Current_Bearing is None:
        Current_Bearing = heading 

    Current_Bearing = float(Current_Bearing)
    # log_and_print(f"Current Bearing : {Current_Bearing:.1f} , Target Bearing : {target_bearing:.1f}")
    
    bearing_diff = Current_Bearing - target_bearing
    
    # normalize bearing difference to range between -180 and 180 degrees
    if bearing_diff < -180:
        bearing_diff += 360
    elif bearing_diff > 180:
        bearing_diff -= 360 

    if abs(bearing_diff) < 1:           #Nullify the small the bearing difference
        temp = bearing_diff
        bearing_diff = 0
        STEER_GAIN = 300
        # log_and_print(f"    Nullied | Bearing Difference of {temp:.1f}")
    elif abs(bearing_diff) > 20:
        STEER_GAIN = 1200
    else:
        log_and_print(f"    Bearing Difference : {bearing_diff:.1f}")

    steer_output = STEER_GAIN * np.arctan(-1 * 2 * 3.5 * np.sin(np.radians(bearing_diff)) / 8)
    return steer_output, bearing_diff

def calculate_bearing_difference_for_speed_reduction(currentLocation, Current_Bearing):
    global wp
    RAD_TO_DEG_CONVERSION = 57.2957795
    next_wp = 4                 # changed from 3 to 4
    
    if wp+next_wp < wp_len:
        off_y = - currentLocation[0] + waypoints[wp+next_wp][0]
        off_x = - currentLocation[1] + waypoints[wp+next_wp][1]
    else:
        off_y = - currentLocation[0] + waypoints[wp][0]
        off_x = - currentLocation[1] + waypoints[wp][1]
        
    # calculate bearing based on position error
    target_bearing = 90.00 + math.atan2(-off_y, off_x) * RAD_TO_DEG_CONVERSION

    # convert negative bearings to positive by adding 360 degrees
    if target_bearing < 0:
        target_bearing += 360.00
    
    Current_Bearing = heading 
    while Current_Bearing is None:
        Current_Bearing = heading 

    Current_Bearing = float(Current_Bearing)
    future_bearing_diff = Current_Bearing - target_bearing
    
    # normalize bearing difference to range between -180 and 180 degrees
    if future_bearing_diff < -180:
        future_bearing_diff += 360
    elif future_bearing_diff > 180:
        future_bearing_diff -= 360 

    # log_and_print(f"    Future Bearing Difference for ({next_wp}) waypoint : {future_bearing_diff:.1f}")
    return future_bearing_diff

def navigation_output(latitude, longitude, Current_Bearing):
    global counter, speed, wp, CW_flag, saw_pothole, const_speed, pothole_flag, acc_z, frame_count
    flasher = 3                     # 0 None, 1 Left, 2 Right, 3 Both ; For Indicator
    counter = (counter + 1) % 256

    # log_and_print(f"Current :- Latitude: {latitude} , Longitude: {longitude}")
    log_and_print(f"2D Standard Deviation(in cms): {100*lat_delta:.2f} cm") 

    currentLocation = [latitude, longitude]
    distance_to_final_waypoint = np.linalg.norm(np.array(currentLocation) - waypoints[-1]) * LAT_LNG_TO_METER

    if (distance_to_final_waypoint> 1 and wp < wp_len):     # to check if the final point is not less than 1m
        
        if CW_flag==1:
            log_and_print(f"Collision Warning Status : Caution")
        elif CW_flag==2:
            log_and_print(f"Collision Warning Status : Brake Signal") 
        else:
            log_and_print(f"Collision Warning Status : Safe")
            
        steer_output, bearing_diff = calculate_steer_output(currentLocation, Current_Bearing)
        steer_output *= -1.0

        future_bearing_diff = calculate_bearing_difference_for_speed_reduction(currentLocation, Current_Bearing)
        # next_bearing_diff = bearing_diff - future_bearing_diff
        # log_and_print(f"Future & Current Bearing diff : {next_bearing_diff:.1f}")
        
        const_speed = speed

        if wp < 1 or wp_len < wp:                 #Slow start and end in the waypoints
            const_speed = turning_factor*speed

        if(abs(bearing_diff)>5):
            const_speed = turning_factor*speed
            log_and_print(f"Turning Speed from code : {const_speed:.0f} kmph")
        elif(abs(future_bearing_diff) > 5):
            const_speed = 0.8*speed
            log_and_print(f"Curve Speed from code : {const_speed:.0f} kmph")
        
        ########### Collision WARNING 
        if (CW_flag == 1):             # Caution Flag
            const_speed = turning_factor*speed
        elif(CW_flag == 2):            # Brake Flag
            const_speed = 0

        ####### speed bump and pothole  
        # if(pothole_flag == 1):         #saw pothole/ speedbreaker
        #     frame_count = 0
        #     const_speed = pothole_speed
        # elif(saw_pothole == 1 and pothole_flag == 0):       # pothole not visible now, but under the hood
        #     frame_count += 1                            # counting frames for 70
        #     if(frame_count < 70):
        #         const_speed= pothole_speed
        #         # log_and_print(f" Speedbump Frame Count : {frame_count}")
        #     else:
        #         saw_pothole = 0
        #         const_speed = speed  
            
        # if(pothole_flag == 1):       # Pothole just crossed from the vision but will the vehicle
        #     saw_pothole = pothole_flag
    
        # if pothole_flag == 1 or saw_pothole == 1:
        #     print(f"SpeedBump Detected")
        
        #####################Testing for single speedbump
        if(pothole_flag == 1):         #saw pothole/ speedbreaker
            frame_count = 0
            const_speed = pothole_speed
        elif(saw_pothole == 1 and pothole_flag == 0):       # pothole not visible now, but under the hood
            if acc_z > 12:
                saw_pothole = 0
                const_speed = speed
                print(" 1222222222222222222222222222222222222222222222222222222")
            
            frame_count += 1                            # counting frames for 70
            if(frame_count > 100):
                saw_pothole = 0
                const_speed = speed
                print("                                                                     False Detection of Speedbump")
        
        print(f"Acceleration : {acc_z:.1f}")
                
            
        if(pothole_flag == 1):       # Pothole just crossed from the vision but will the vehicle
            saw_pothole = pothole_flag
    
        if pothole_flag == 1 or saw_pothole == 1:
            log_and_print(f"SpeedBump Detected via present:{pothole_flag}  past:{saw_pothole}")

        ## Testing ends here


        ############Failed Testing for Multiple Speedbump
        # if pothole_flag == 1:  # Saw pothole/speedbreaker
        #     const_speed = pothole_speed
        # elif saw_pothole == 1 and pothole_flag == 0:  # Pothole not visible now, but under the hood
        #     print("Inside the master ")
        #     pothole_time = pothole_time + 1
        #     pothole_start_time = None  # Initialize pothole_start_time here
        #     if acc_z > 10.5:
        #         pothole_start_time = pothole_start_time + 1
        #         const_speed = pothole_speed
        #         log_and_print(f"Waiting for the Speedbump {acc_z:.1f}")
        #     try:
        #         print("                              inside try")
        #         if pothole_start_time is not None:
        #             print("                                     inside try-if")
        #             p_duration = pothole_time - pothole_start_time
        #             if p_duration > pothole_duration:
        #                 saw_pothole = 0
        #                 log_and_print("Pothole passed: Normal Speed Applied")
        #     except Exception as e:
        #         if time.time() - max_pothole_time > pothole_duration + 2:
        #             saw_pothole = 0
        #             log_and_print("False pothole detected: Normal Speed Applied")
        #     const_speed = pothole_speed
        # if(pothole_flag == 1):       # Pothole just crossed from the vision but will the vehicle
        #     saw_pothole = pothole_flag
    
        # if pothole_flag == 1 or saw_pothole == 1:
        #     log_and_print(f"SpeedBump Detected via present:{pothole_flag}  past:{saw_pothole}")

        # log_and_print(f"Acceleraton {acc_z:.1f}")
        # #############
        ########################################

        distance_to_nextpoint = np.linalg.norm(np.array(currentLocation) - waypoints[wp]) * LAT_LNG_TO_METER
        log_and_print(f"{wp} out of {wp_len} | Next Coordinate distance : {distance_to_nextpoint:.1f} m")           # For Testing
        try:
            if wp < wp_len and distance_to_nextpoint < LOOK_AHEAD_DISTANCE:
                wp += 1
        except IndexError:
            log_and_print("Waypoint index out of range! - Seems like you are at wrong location or Inputted wrong waypoint")
    else:
        log_and_print("----- FINISHED  -----")
        log_and_print("Brake Activated")
        steer_output = 0
        const_speed = 0
        flasher = 0
    
    log_and_print(f"Speed set by code: {const_speed:.0f} kmph")
    try:
        message = send_message_to_mabx(const_speed, steer_output, 0, flasher, counter)  
        mabx_socket.sendto(message, mabx_addr)
    except Exception as e:
        log_and_print(f"Error sending message to MABX: {e}")

def mainLoop(): 
    while not rospy.is_shutdown():
        try:
            log_and_print(f"Current Coordinate No. : {wp}")
            log_and_print(" ")
            #log_and_print(f"Velocity in kmph as per GNSS= {current_vel:.0f} kmph")
            
            latitude = float(lat)
            longitude = float(lng)
            Current_Bearing = float(heading)

            # time.sleep(SLEEP_INTERVAL/1000)
            navigation_output(latitude, longitude, Current_Bearing)
            time.sleep(SLEEP_INTERVAL/1000)           
        except KeyboardInterrupt:       #Currently not working
            log_and_print("Autonomous Mode is terminated manually!")
            message = send_message_to_mabx(0, 0, 0, 0, counter)     #0
            mabx_socket.sendto(message, mabx_addr)  
            raise SystemExit
        
        except Exception as e:
            log_and_print(f"An error occurred: {e}")

if __name__ == '__main__':
    
    global speed, reduction_factor, steer_output, counter, wp, file_path, pothole_flag, pothole_speed

    # Define the path(not relative path) to the waypoints file
    # file_path = '/home/intel-nuc/Desktop/Solio-ADAS/Solio-Suzuki/navigation/solio-waypoints/Solio1/waypoints-speedtest-reverse.txt'
    file_path = '/usr/local/zed/samples/object-avoidance-zed-suzuki/waypoints_2023-12-16-11:36:52.txt'
    
    log_dir = "devLogs"
    logger = setup_logging(log_dir, file_path)
    logger.info('Development Code Starting')

    # Set sleep interval and lookahead distance
    SLEEP_INTERVAL = 100            # CHANGED FROM 5 TO 100
    LOOK_AHEAD_DISTANCE = 3

    # Define initial speeds, pothole_speed, turning_factor
    speed = 10
    pothole_speed = 10
    turning_factor = 0.6
    wp = 0
    steer_output = 0
    counter = 0
    
    global saw_pothole, frame_count
    saw_pothole = 0
    frame_count = 0
    
    # Get the list of waypoints from the file
    waypoints = get_coordinates(file_path)
    wp_len = len(waypoints)
    
    logger.info(f" Speed : {speed} , Pothole Speed : {pothole_speed}, Turning Factor : {turning_factor} , Sleep : {SLEEP_INTERVAL}, Look ahead distance : {LOOK_AHEAD_DISTANCE}")       # Test SteerGAIN here
    # Start the main loop
    mainLoop()