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
    print(f"Actuation command send to MABX = {str(message)}")

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
    distance_to_nextpoint = np.linalg.norm(np.array(current_location) - WAYPOINTS[WP_LEN - 1]) * const.LAT_LNG_TO_METER
    print(f"Distance between last and current waypoint = {distance_to_nextpoint} m")
    return distance_to_nextpoint

def has_reached_end():
    global wp
    return wp >= WP_LEN
    
def navigation_output(latitude, longitude, current_bearing):
    global wp, collision_warning, lane_state, collision_avoidance, overtake_lat_lon, optical_flow, current_vel, heading  
    
    if has_reached_end():
        while True:
            actuate(const.BRAKE_SPEED, const.BEARING_ZERO)
            print("Reached destination !!!!!!!!!!!!!!")

    current_location = [latitude, longitude]
    collision_warning = int(collision_warning)
    
    if (int(current_vel) == const.BRAKE_SPEED and collision_warning == const.URGENT_WARNING) or lane_state != const.DRIVING_LANE:
        print("Entering collision or lane change segment")
        print(
            f"collision_warning = {const.STATE_DICT[collision_warning]}, "
            f"collision_avoidance = {const.STATE_DICT[collision_avoidance]}, "
            f"Lane_state = {const.STATE_DICT[lane_state]}, "
            f"BESTVEL = {current_vel}, "
            f"HEADING = {heading}, "
            f"BESTPOS = {current_location}, "
            f"overtake_lat_lon = {overtake_lat_lon}, "
            f"optical_flow = {optical_flow}")

        if collision_avoidance == const.OVERTAKE or lane_state == const.CHANGE_LANE:
            print("Entering overtake or lane change sub-segment")
            if lane_state == const.DRIVING_LANE and (optical_flow == const.SAFE_TO_OVERTAKE or optical_flow == const.TRAFFIC_FROM_LEFT):
                _dist, _lat, _lon = overtake_lat_lon.split(",")
                overtake_location = [float(_lat), float(_lon)]
                print(f"Going for overtake at distance {_dist} and lat: {_lat} , lon:{_lon}")               

                if not util.has_reached(current_location, overtake_location):
                    lane_state = const.CHANGE_LANE
                    lane_state_publish.publish(lane_state)
                    steer_output, bearing_diff = util.calculate_steer_output_change_lane(current_location, overtake_location, current_bearing, heading)
                    steer_output = steer_output * -1.00
                    actuate(const.CHANGE_SPEED, current_bearing, steer_output)
                else:
                    lane_state = const.OVERTAKE_LANE
                    lane_state_publish.publish(lane_state)
            elif lane_state == const.CHANGE_LANE:
                print("Entering lane change sub-sub-segment")
                _dist, _lat, _lon = overtake_lat_lon.split(",")
                overtake_location = [float(_lat), float(_lon)]
                if not util.has_reached(current_location, overtake_location):
                    lane_state = const.CHANGE_LANE
                    lane_state_publish.publish(lane_state)
                    steer_output, bearing_diff = util.calculate_steer_output_change_lane(current_location, overtake_location, current_bearing, heading)
                    steer_output = steer_output * -1.00
                    actuate(const.CHANGE_SPEED, current_bearing, steer_output)
                else:
                    lane_state = const.OVERTAKE_LANE
                    lane_state_publish.publish(lane_state)
            else:
                print(f"$$$$ Lane State = {lane_state} $$$$")      
        elif lane_state == const.OVERTAKE_LANE:
            print("Entering overtake lane sub-segment")
            _next_lat, _next_lon = util.get_next_overtake_waypoint(current_location[0], current_location[1])
            if collision_warning == const.URGENT_WARNING:
                actuate(const.BRAKE_SPEED, current_bearing, steer_output) 
            else:
                next_location = [_next_lat, _next_lon]
                steer_output, bearing_diff = util.calculate_steer_output_change_lane(current_location, next_location, current_bearing, heading)
                steer_output = steer_output * -1.00
                actuate(const.OVERTAKE_SPEED, current_bearing, steer_output) 
        else:
            print(f"$$$$ Lane State = {lane_state} $$$$")

    else:
        print("Entering normal driving segment")
        print(
            f"collision_warning = {const.STATE_DICT[collision_warning]}, "
            f"collision_avoidance = {const.STATE_DICT[collision_avoidance]}, "
            f"Lane_state = {const.STATE_DICT[lane_state]}, "
            f"BESTVEL = {current_vel}, "
            f"HEADING = {heading}, "
            f"BESTPOS = {current_location}, "
            f"overtake_lat_lon = {overtake_lat_lon}, "
            f"optical_flow = {optical_flow}")
        steer_output, bearing_diff = calculate_steer_output(current_location, current_bearing)
        steer_output = steer_output * -1.00
        const_speed = util.get_speed(collision_warning, lane_state, bearing_diff)
        actuate(const_speed, current_bearing, steer_output)
        if get_distance_to_next_waypoint(current_location) < const.WP_DIST:
            wp = wp + 1

if __name__ == '__main__':    
    global counter, wp
    
    wp = 0
    counter = 0
    
    while not rospy.is_shutdown():
        try:
            print("############## Navigation #############")      
            latitude = float(lat)
            longitude = float(lng)
            current_bearing = float(heading)
            time.sleep(0.1)
            navigation_output(latitude, longitude, current_bearing)
            lane_state_publish.publish(lane_state)
            print(f"Current waypoint # = {wp}")
            print(f"Ego vehicle velocity = {float(current_vel)} kmph")  
            print("#####################################\n\n")           
        except Exception as e:
            print(f"Inside rospy.is_shutdown() loop: {str(e)}")
            pass
        
        
