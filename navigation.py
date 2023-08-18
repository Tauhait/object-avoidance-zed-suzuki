from calendar import day_abbr
import socket
import threading
import select
import time, math
import numpy as np
import pandas as pd
from collections import deque
import rospy
from novatel_oem7_msgs.msg import BESTPOS, BESTVEL, INSPVA
from std_msgs.msg import Float32, String
from signal import signal, SIGPIPE, SIG_DFL 

TOP_SPEED = 12
MABX_IP = "192.168.50.1"  # Mabx IP for sending from Pegasus
MABX_PORT = 30000  # Mabx port for sending from Pegasus
# global lat, lng, collision_avoidance, optical_flow, streering_angle
global lat, lng
BUFFER_SIZE = 4096  # packet size
LOCAL_INTERFACE = 'eno1'
NAVIGATION_DATA = 'data.csv'
MABX_SOCKET = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
WAYPOINT_FILENAME = '/usr/local/zed/samples/object-avoidance-zed-suzuki/new-waypoints_2023-08-18-15:00:40.txt'
MABX_ADDR = (MABX_IP, MABX_PORT)
STEER_GAIN = 1200          # For Tight Turns 1200 can be used
# WAYPOINTS = pd.read_csv(WAYPOINT_FILENAME, dtype='str')
# WAYPOINTS = [[np.float64(val) for val in row] for row in WAYPOINTS.values.tolist()]
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
WAYPOINTS = get_coordinates(WAYPOINT_FILENAME)
TURN_BEARNG_THRESHOLD = 6
WP_LEN = len(WAYPOINTS)
# Conversion factor for latitude and longitude to meters
LAT_LNG_TO_METER = 1.111395e5

WP_DIST = 3
RAD_TO_DEG_CONVERSION = 57.2957795

rospy.init_node('navigation', anonymous=True)

######################### Perception

def callback_collision_warning(data):
    global collision_warning
    collision_warning = 0
    collision_warning = data.data
    
rospy.Subscriber("/collision_warning", Float32, callback_collision_warning)

def callback_collision_avoidance(data):
    global collision_avoidance
    collision_avoidance = 'CONTINUE'
    collision_avoidance = data.data
    
rospy.Subscriber("/collision_avoidance", String, callback_collision_avoidance)

def callback_streering_angle(data):
    global streering_angle
    streering_angle = 0
    streering_angle = data.data
    
rospy.Subscriber("/streering_angle", Float32, callback_streering_angle)

def callback_optical_flow(data):
    global optical_flow
    optical_flow = 'SAFE-TO-OVERTAKE'
    optical_flow = data.data
    
rospy.Subscriber("/optical_flow", String, callback_optical_flow)

######################### Perception

######################### GNSS    
def callback_vel(data):
    global current_vel 
    current_vel = 3.6 * data.hor_speed  
    
rospy.Subscriber("/novatel/oem7/bestvel",BESTVEL, callback_vel)

def callback_heading(data):
    global heading
    heading = data.azimuth  
    
rospy.Subscriber("/novatel/oem7/inspva",INSPVA, callback_heading)

def callback_latlong(data):
    global lat,lng   
    lat = data.lat
    lng = data.lon
    
rospy.Subscriber("/novatel/oem7/bestpos",BESTPOS, callback_latlong)
time.sleep(0.1)
print("lat, lng: {}, {}".format(lat, lng))

######################### GNSS 


######################### mabx

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

################################ mabx

######################### Navigation
def calculate_steer_output(currentLocation, current_bearing):
    global wp
    off_y = - currentLocation[0] + WAYPOINTS[wp][0]
    off_x = - currentLocation[1] + WAYPOINTS[wp][1]

    # calculate bearing based on position error
    target_bearing = 90.00 + math.atan2(-off_y, off_x) * RAD_TO_DEG_CONVERSION 

    # convert negative bearings to positive by adding 360 degrees
    if target_bearing < 0:
        target_bearing += 360.00
    
    current_bearing = heading 
    while current_bearing is None:
        current_bearing = heading 
    current_bearing = float(current_bearing)
    print(f"Azimuth: {current_bearing}")
    # calculate the difference between heading and bearing
    bearing_diff = current_bearing - target_bearing
    
    

    # normalize bearing difference to range between -180 and 180 degrees
    if bearing_diff < -180:
        bearing_diff = bearing_diff + 360

    if bearing_diff > 180:
        bearing_diff = bearing_diff - 360 

    print("Diff",bearing_diff)


    steer_output = STEER_GAIN * np.arctan(-1 * 2 * 3.5 * np.sin(np.radians(bearing_diff)) / 8)
    
    return steer_output, bearing_diff

def navigation_output(latitude, longitude, current_bearing):
    global counter, prev_time, curr_time, speed, wp, collision_warning, collision_avoidance, streering_angle, optical_flow, prev, count
    flasher = get_flasher(current_bearing)     # 1 Left, 2 Right, 3 Right ; For Flasher
    counter = (counter + 1) % 256
    const_speed = TOP_SPEED
    print(f"Latitude: {latitude}")
    print(f"Longitude: {longitude}")
    print(f"Azimuth: {current_bearing}")

    currentLocation = [latitude, longitude]

    if ((np.linalg.norm(np.array(currentLocation) - WAYPOINTS[len(WAYPOINTS) - 1]) * LAT_LNG_TO_METER) > 1 and wp < WP_LEN):
    
        print("collision_warning", collision_warning)
        print("collision_avoidance", collision_avoidance) 
        print("optical_flow", optical_flow)  
        print("streering_angle", streering_angle) 
                
        steer_output, bearing_diff = calculate_steer_output(currentLocation, current_bearing)
        steer_output = steer_output * -1.00
        ####### DECIDE SPEED #############
        if(abs(bearing_diff) > TURN_BEARNG_THRESHOLD):   #turning speed
            const_speed = speed - 4
            if(collision_warning == 2):
                const_speed = 0
                # if collision_avoidance == 'OVERTAKE' and optical_flow == 'SAFE-TO-OVERTAKE':
                #     steer_output = streering_angle
                #     const_speed = 4
        else:
            ########### COLLISION WARNING 
            if (collision_warning == 1):            #dynamic zed
                const_speed = 7
           
            elif(collision_warning == 2): 
                const_speed = 0               
                # if collision_avoidance == 'OVERTAKE' and optical_flow == 'SAFE-TO-OVERTAKE':
                #     steer_output = streering_angle
                #     const_speed = 2
                # else :
                #     const_speed = 0
        
        print(f"steering power : {steer_output:.4f}")
        distance_to_nextpoint = np.linalg.norm(np.array(currentLocation) - WAYPOINTS[wp]) * LAT_LNG_TO_METER
        if (wp < WP_LEN and distance_to_nextpoint < WP_DIST):
            wp = wp + 1
            # Append the data to the CSV file
            with open(NAVIGATION_DATA, 'a') as csv_file:
                lat_lng_steering_speed = f"{latitude},{longitude},{steer_output},{speed}\n"
                csv_file.write(lat_lng_steering_speed)
    else:
        print("FINISHED!!!!!!!!!!!!!!")
        steer_output = 0
        const_speed = 0
       
    # print("Set speed", const_speed)
    message = get_msg_to_mabx(const_speed, steer_output, 0, flasher, counter)
    MABX_SOCKET.sendto(message, MABX_ADDR)
    curr_time = time.time()
    diff = round((curr_time-prev_time),4)
    print("time to mabx: ", diff)
    prev_time = curr_time
    
######################### Navigation


if __name__ == '__main__':    
    global speed, steer_output, counter, prev_time, curr_time, wp
    
    speed = TOP_SPEED
    wp = 0
    steer_output = 0
    counter = 0

    global new_file
    prev_time = time.time()
    curr_time = time.time()
    new_file=time.time()
    global prev, count
    prev = 0
    count = 0
        
    while not rospy.is_shutdown():
        try:
            print("#####################################")
            print("vel in kmph = ",float(current_vel))            
            latitude = float(lat)
            longitude = float(lng)
            current_bearing = float(heading)
         
            time.sleep(0.1)
            navigation_output(latitude, longitude, current_bearing)
            print("wp", wp)            
        except False:
            pass
        
        
