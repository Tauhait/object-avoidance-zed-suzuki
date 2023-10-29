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
WAYPOINTS = util.get_coordinates(const.WAYPOINT_FILENAME)
WP_LEN = len(WAYPOINTS)

# global lat, lng
global lat, lng, collision_avoidance, optical_flow, overtake_lat_lon, lane_state, collision_warning

stop_counter = 0

rospy.init_node('navigation', anonymous=True)
lane_state_publish = rospy.Publisher('lane_state', Int8, queue_size=1)


def callback_lane_state(data):
    global lane_state
    lane_state = data.data
    
rospy.Subscriber("/lane_state", Int8, callback_lane_state)

def callback_collision_warning(data):
    global collision_warning
    collision_warning = data.data
    
rospy.Subscriber("/collision_warning", Float32, callback_collision_warning)

def callback_collision_avoidance(data):
    global collision_avoidance
    collision_avoidance = data.data
    
rospy.Subscriber("/collision_avoidance", Int8, callback_collision_avoidance)

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
time.sleep(0.1)

def calculate_steer_output(currentLocation, current_bearing):
    global wp
    off_y = - currentLocation[0] + WAYPOINTS[wp][0]
    off_x = - currentLocation[1] + WAYPOINTS[wp][1]

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

def navigation_output(latitude, longitude, current_bearing):
    global counter, prev_time, curr_time, speed, wp, collision_warning, lane_state, collision_avoidance, overtake_lat_lon, optical_flow, prev, count, stop_counter
    flasher = util.get_flasher(current_bearing)     # 1 Left, 2 Right, 3 Right ; For Flasher
    counter = (counter + 1) % 256
    const_speed = const.TOP_SPEED
    print(f"Latitude: {latitude}")
    print(f"Longitude: {longitude}")
    print(f"Azimuth: {current_bearing}")

    currentLocation = [latitude, longitude]
    print(f"Distance between last and currwent waypoint = {np.linalg.norm(np.array(currentLocation) - WAYPOINTS[WP_LEN - 1]) * const.LAT_LNG_TO_METER} m")
    if lane_state == const.DRIVING_LANE:
        print(f"Navigating in DRIVING_LANE")
        if (np.linalg.norm(np.array(currentLocation) - WAYPOINTS[WP_LEN - 1]) * const.LAT_LNG_TO_METER > 1 and wp < WP_LEN):
            steer_output, bearing_diff = calculate_steer_output(currentLocation, current_bearing)
            steer_output = steer_output * -1.00
            
            # print(f"wp = {wp}, steer_output = {steer_output}, const_speed = {const_speed}, bearing_diff = {bearing_diff}, stop_counter = {stop_counter}")
            ####### DECIDE SPEED #############
            const_speed = util.get_speed(collision_warning, lane_state, bearing_diff)
            # print(f"optical_flow = {optical_flow}")
            # print(f"wp = {wp}, steer_output = {steer_output}, const_speed = {const_speed}, bearing_diff = {bearing_diff}, stop_counter = {stop_counter}")
            # print(f"wp = {wp}, steer_output = {steer_output}, const_speed = {const_speed}, bearing_diff = {bearing_diff}")
            # stop_counter = stop_counter + 1
            # print(f"const_speed = {const_speed}, optical_flow = {optical_flow}")
            # print(f"const.BRAKE_SPEED = {const.BRAKE_SPEED}, const.SAFE_TO_OVERTAKE = {const.SAFE_TO_OVERTAKE}")
            if const_speed == const.BRAKE_SPEED and optical_flow == const.SAFE_TO_OVERTAKE:
                stop_counter = stop_counter + 1
            
            # print(f"wp = {wp}, steer_output = {steer_output}, const_speed = {const_speed}, bearing_diff = {bearing_diff}, stop_counter = {stop_counter}")
            print(f"collision_avoidance = {collision_avoidance}, WP_LEN = {WP_LEN}")
            if stop_counter > const.WAIT_TIME and collision_avoidance == const.OVERTAKE and optical_flow == const.SAFE_TO_OVERTAKE:
                print(f"Waited for {stop_counter} ms so navigating to CHANGE_LANE")
                lane_state = const.CHANGE_LANE
                stop_counter = 0
            distance_to_nextpoint = np.linalg.norm(np.array(currentLocation) - WAYPOINTS[wp]) * const.LAT_LNG_TO_METER
            if wp < WP_LEN and distance_to_nextpoint < const.WP_DIST:
                wp = wp + 1
            print(f"wp = {wp}, steer_output = {steer_output}, const_speed = {const_speed}")
        else:
            const_speed = const.BRAKE_SPEED
            print("FINISHED!!!!!!!!!!!!!!")
        
        print(f"wp = {wp}, steer_output = {steer_output}, const_speed = {const_speed}, bearing_diff = {bearing_diff}, stop_counter = {stop_counter}")

    elif lane_state == const.CHANGE_LANE:
        print(f"Navigating in CHANGE_LANE")
        const_speed = util.get_speed(collision_warning, lane_state, bearing_diff)
        _dist, _lat, _lon = overtake_lat_lon.split(",")
        overtake_location = [float(_lat), float(_lon)]
        if util.has_reached(currentLocation, overtake_location):
            print(f"Reached overtake waypoint, changing lane_state to OVERTAKE_LANE")
            lane_state = const.OVERTAKE_LANE
    else :
        print(f"Navigating in OVERTAKE_LANE")
        const_speed = util.get_speed(collision_warning, lane_state, bearing_diff)
        if collision_avoidance == const.SWITCH:
            # go back to original lane
            print(f"going back to original lane")
            distance_to_nextpoint = np.linalg.norm(np.array(currentLocation) - WAYPOINTS[wp]) * const.LAT_LNG_TO_METER
            while wp < WP_LEN and distance_to_nextpoint < const.WP_DIST:
                wp = wp + 1
                distance_to_nextpoint = np.linalg.norm(np.array(currentLocation) - WAYPOINTS[wp]) * const.LAT_LNG_TO_METER
            lane_state = const.DRIVING_LANE
        else:
            next_lat, next_lon = util.get_next_overtake_waypoint(currentLocation[0], currentLocation[1])
            print(f"Next overtake lane state waypoint {next_lat}, {next_lon}")
            next_loc = [next_lat, next_lon]
            steer_output, bearing_diff = calculate_steer_output(next_loc, current_bearing)
            steer_output = steer_output * -1.00

    print(f"Speed:  {const_speed}")
    print(f"Steering power: {steer_output:.4f}")
    message = util.get_msg_to_mabx(const_speed, steer_output, 0, flasher, counter)
    MABX_SOCKET.sendto(message, MABX_ADDR)
    print(f"MABX socket message send")


if __name__ == '__main__':    
    global speed, steer_output, counter, prev_time, curr_time, wp
    
    speed = const.TOP_SPEED
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
    lane_state_publish.publish(const.DRIVING_LANE)
    while not rospy.is_shutdown():
        try:
            print("#####################################")
            # print("vel in kmph = ",float(current_vel))   
            lane_state_publish.publish(lane_state)     
            latitude = float(lat)
            longitude = float(lng)
            current_bearing = float(heading)
            time.sleep(0.1)
            navigation_output(latitude, longitude, current_bearing)
            print("wp", wp)            
        except Exception:
            pass
        
        
