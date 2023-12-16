import util
import constant as const
import socket
import threading
import select
import time, math
import numpy as np
import pandas as pd
import rospy
from calendar import day_abbr
from collections import deque
from novatel_oem7_msgs.msg import BESTPOS, BESTVEL, INSPVA
from std_msgs.msg import Float32, String, Int8, Int32
from signal import signal, SIGPIPE, SIG_DFL 

MABX_SOCKET = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
MABX_ADDR = (const.MABX_IP, const.MABX_PORT)
OVERTAKE_WAYPOINTS = util.get_coordinates(const.OVERTAKE_WAYPOINT_FILENAME)
WAYPOINTS = util.get_coordinates(const.WAYPOINT_FILENAME)
WP_LEN = len(WAYPOINTS)
OVERTAKE_WP_LEN = len(OVERTAKE_WAYPOINTS)

global lat, lng, collision_avoidance, optical_flow, overtake_lat_lon, lane_state, collision_warning

rospy.init_node('navigation', anonymous=True)
lane_state_publish = rospy.Publisher('lane_state', Int32, queue_size=1)


def callback_lane_state(data):
    global lane_state
    lane_state = data.data
    
rospy.Subscriber("/lane_state", Int32, callback_lane_state)

def callback_collision_warning(data):
    global collision_warning
    collision_warning = data.data
    
rospy.Subscriber("/collision_warning", Int32, callback_collision_warning)

def callback_collision_avoidance(data):
    global collision_avoidance
    collision_avoidance = data.data
    
rospy.Subscriber("/collision_avoidance", Int32, callback_collision_avoidance)

def callback_streering_angle(data):
    global overtake_lat_lon
    overtake_lat_lon = '0,0,0'
    overtake_lat_lon = data.data
    
rospy.Subscriber("/overtake_lat_lon", String, callback_streering_angle)

def callback_optical_flow(data):
    global optical_flow
    optical_flow = data.data
    
rospy.Subscriber("/optical_flow", Int32, callback_optical_flow)
   
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

def actuate(const_speed, current_bearing, steer_output):
    global counter
    flasher = util.get_flasher(current_bearing) # 1 Left, 2 Right, 3 Right ; For Flasher
    counter = (counter + 1) % 256
    message = util.get_msg_to_mabx(const_speed, steer_output, 0, flasher, counter)
    MABX_SOCKET.sendto(message, MABX_ADDR)
    # print(f"Actuation command send to MABX = {str(message)}")

def overtake_calculate_steer_output(current_location, current_bearing):
    global wp_overtake
    off_y = - current_location[0] + OVERTAKE_WAYPOINTS[wp_overtake][0]
    off_x = - current_location[1] + OVERTAKE_WAYPOINTS[wp_overtake][1]

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

def calculate_steer_output(current_location, current_bearing):
    global wp
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
        bearing_diff = bearing_diff + 360

    if bearing_diff > 180:
        bearing_diff = bearing_diff - 360 

    steer_output = const.STEER_GAIN * np.arctan(-1 * 2 * 3.5 * np.sin(np.radians(bearing_diff)) / 8)
    
    return steer_output, bearing_diff

def get_distance_to_next_waypoint(current_location):
    # distance_to_nextpoint = np.linalg.norm(np.array(current_location) - WAYPOINTS[wp + 1]) * const.LAT_LNG_TO_METER
    distance_to_nextpoint = 0
    if wp < WP_LEN:
        distance_to_nextpoint = np.linalg.norm(np.array(current_location) - WAYPOINTS[wp]) * const.LAT_LNG_TO_METER
    print(f"Distance between next waypoint and current location = {distance_to_nextpoint} m")
    return distance_to_nextpoint

def overtake_get_distance_to_next_waypoint(current_location):
    # distance_to_nextpoint = np.linalg.norm(np.array(current_location) - WAYPOINTS[wp + 1]) * const.LAT_LNG_TO_METER
    distance_to_nextpoint = np.linalg.norm(np.array(current_location) - OVERTAKE_WAYPOINTS[wp_overtake]) * const.LAT_LNG_TO_METER
    print(f"Distance between next waypoint and current location = {distance_to_nextpoint} m")
    return distance_to_nextpoint

def has_reached_end():
    global wp
    return wp >= WP_LEN
    
def navigation_output(latitude, longitude, current_bearing):
    global wp, collision_warning, lane_state, collision_avoidance, overtake_lat_lon, optical_flow, current_vel, heading, some_var, overtake_location, wp_overtake
    
    current_location = [latitude, longitude]
    collision_warning = int(collision_warning)
    print("===============================================================")
    print("===============================================================")
    print(f"Waypoint # {wp}\n")
    print(
        f"Collision Warning = {const.STATE_DICT[collision_warning]},\n"
        f"Collision Avoidance = {const.STATE_DICT[collision_avoidance]},\n"
        f"Lane State = {const.STATE_DICT[lane_state]},\n"
        f"Velocity = {current_vel},\n"
        f"Heading = {heading},\n"
        f"Current Location = {current_location},\n"
        # f"Overtake Location = {overtake_lat_lon},\n"
        f"Optical Flow = {const.STATE_DICT[optical_flow]}\n")
    print("===============================================================")
    print("===============================================================")
    local_speed = const.BRAKE_SPEED

    if np.linalg.norm(np.array(current_location) - WAYPOINTS[WP_LEN - 1]) * const.LAT_LNG_TO_METER > 1 and wp < WP_LEN:
        if some_var == 0 and lane_state == const.DRIVING_LANE:
            print("NORMAL")
            steer_output, bearing_diff = calculate_steer_output(current_location, current_bearing)
            steer_output = steer_output * -1.00
            const_speed = util.get_speed(collision_warning, lane_state, bearing_diff)
            if some_var == 0 and const_speed == 0:
                some_var = 1
                lane_state = const.OVERTAKE_LANE
                lane_state_publish.publish(lane_state)
                local_speed = const.BRAKE_SPEED
                # actuate(const.BRAKE_SPEED, current_bearing, steer_output)
            else:
                local_speed = const_speed
                # actuate(const_speed, current_bearing, steer_output)
            if get_distance_to_next_waypoint(current_location) < const.WP_DIST:
                wp = wp + 1
        
        else:
            print("OVERTAKING")
            print(f"\nsome_var = {some_var}\n")
            if wp_overtake >= OVERTAKE_WP_LEN:
                some_var = 0
                lane_state = const.DRIVING_LANE
                lane_state_publish.publish(lane_state)
            else:
                steer_output, bearing_diff = overtake_calculate_steer_output(current_location, current_bearing)
                steer_output = steer_output * -1.00
                local_speed = const.OVERTAKE_SPEED
                # actuate(const.OVERTAKE_SPEED, current_bearing, steer_output)
                if overtake_get_distance_to_next_waypoint(current_location) < const.WP_DIST:
                    wp_overtake = wp_overtake + 1
        
        actuate(local_speed, current_bearing, steer_output)
    else:
        print("Reached destination !!!!!!!!!!!!!!")
        actuate(const.BRAKE_SPEED, current_bearing, const.ZERO_STEET_OUTPUT)


if __name__ == '__main__':    
    global counter, wp, some_var, overtake_location, wp_overtake
    
    wp = 0
    wp_overtake = 0
    counter = 0
    # navigation_path = []
    overtake_location = None
    some_var = 0
    
    while not rospy.is_shutdown():
        try:
            print("############## Navigation #############")      
            latitude = float(lat)
            longitude = float(lng)
            current_bearing = float(heading)
            navigation_output(latitude, longitude, current_bearing)
            lane_state_publish.publish(lane_state)
            print("#####################################\n\n")  
        except Exception as e:
            print(f"Inside rospy.is_shutdown() loop: {str(e)}")
            pass
        
        
