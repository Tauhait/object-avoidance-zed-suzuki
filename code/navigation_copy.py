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
    global wp, collision_warning, lane_state, collision_avoidance, overtake_lat_lon, optical_flow, current_vel, heading, some_var, overtake_location
    
    if has_reached_end():
        while True:
            print("Reached destination !!!!!!!!!!!!!!")
            actuate(const.BRAKE_SPEED, current_bearing, const.ZERO_STEET_OUTPUT)
            
    
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

    # ## Handle URGENT CASE
    # if collision_warning == const.URGENT_WARNING:
    #     actuate(const.BRAKE_SPEED, const.BEARING_ZERO, const.ZERO_STEET_OUTPUT)
    
    if some_var == 1 and lane_state == const.OVERTAKE_LANE:
        print(f"$$$$ Lane State = {const.STATE_DICT[lane_state]} $$$$") 
        print(f"current_location = {current_location}, OVERTAKE_WAYPOINTS[wp_overtake] = {OVERTAKE_WAYPOINTS[wp_overtake]}")
        current_location = [latitude, longitude]
        steer_output, bearing_diff = util.overtake_calculate_steer_output(current_location, OVERTAKE_WAYPOINTS[wp_overtake])
        steer_output = steer_output * -1.00
        actuate(const.OVERTAKE_SPEED, current_bearing, steer_output)
        if overtake_get_distance_to_next_waypoint < const.WP_DIST:
            wp_overtake = wp_overtake + 1


    # if some_var == 1 and lane_state == const.CHANGE_LANE and overtake_location is not None:
    #     print(f"\n\n\n\n@@@ some_var = {some_var}\n\n\n")
    #     print(f"Not Yet Reached Overtake Location")
    #     current_location = [latitude, longitude]
    #     print(f"\n\ncurrent_location = {current_location}, overtake_location = {overtake_location}")
    #     steer_output, bearing_diff = util.calculate_steer_output_change_lane(current_location, overtake_location, current_bearing, heading)
    #     steer_output = steer_output * -1.00
    #     actuate(const.CHANGE_SPEED, current_bearing, steer_output)
    #     if util.has_reached(current_location, overtake_location):
    #         some_var = 0
    #         print(f"\n\n\n\n@@@ some_var = {some_var}\n\n\n")
    #         print(f"Reached Overtake Location, Lane State is set to OVERTAKE")
    #         lane_state = const.OVERTAKE_LANE # Change Lane State from CHANGE to OVERTAKE
    #         lane_state_publish.publish(lane_state)
    #         actuate(const.BRAKE_SPEED, current_bearing, steer_output)

    
    # LANE CHANGE
    elif int(current_vel) == const.BRAKE_SPEED and collision_warning == const.URGENT_WARNING and collision_avoidance == const.OVERTAKE and lane_state == const.CHANGE_LANE:
        print("START: Entering lane change segment")
        ## 2 sec section to take decision: START
        curr_time = time.time()
        ca_o = 0
        ls_cl = 0
        of_so = 0
        while time.time() - curr_time < 1:
            # critical wait section
            if collision_avoidance == const.OVERTAKE:
                ca_o += 1
            if lane_state == const.CHANGE_LANE:
                ls_cl += 1
            if optical_flow == const.SAFE_TO_OVERTAKE:
                of_so += 1
        ## 2 sec section to take decision: END
        print(f"ca_o = {ca_o}, ls_cl = {ls_cl}, of_so = {of_so}")

        if ls_cl > const.DECISION_THRESHOLD and ca_o > const.DECISION_THRESHOLD and of_so > const.DECISION_THRESHOLD:
            print("Entering lane change sub-segment")
        #if lane_state == const.DRIVING_LANE and (optical_flow == const.SAFE_TO_OVERTAKE or optical_flow == const.TRAFFIC_FROM_LEFT):
            _dist, _lat, _lon = overtake_lat_lon.split(",")
            overtake_location = [float(_lat), float(_lon)]
            print(f"Going for overtake at distance {_dist} and lat: {_lat} , lon:{_lon}")
            print(f"\n\n\n\n@@@ some_var = {some_var}\n\n\n")
            lane_state = const.OVERTAKE_LANE
            lane_state_publish.publish(lane_state)
            print(f"$$$$ Lane State = {const.STATE_DICT[lane_state]} $$$$")
            time.sleep(1)
            some_var = 1
            # curr_time = time.time()
            # while time.time() - curr_time < 5:
            #     actuate(const.BRAKE_SPEED, current_bearing, steer_output)


            # while not util.has_reached(current_location, overtake_location):
            #     # lane_state = const.CHANGE_LANE
            #     # lane_state_publish.publish(lane_state)
            #     print(f"Not Yet Reached Overtake Location")
            #     current_location = [latitude, longitude]
            #     print(f"\n\ncurrent_location = {current_location}, overtake_location = {overtake_location}")
            #     steer_output, bearing_diff = util.calculate_steer_output_change_lane(current_location, overtake_location, current_bearing, heading)
            #     steer_output = steer_output * -1.00
            #     # if collision_warning == const.URGENT_WARNING:
            #     #     actuate(const.BRAKE_SPEED, current_bearing, steer_output)
            #     #     time.sleep(1)
            #     # else:
            #     # print(f"current_bearing = {current_bearing}, steer_output = {steer_output}, current_vel = {current_vel}")
            #     actuate(const.CHANGE_SPEED, current_bearing, steer_output)
            
            # print(f"Reached Overtake Location, Lane State is set to OVERTAKE")
            # lane_state = const.OVERTAKE_LANE # Change Lane State from CHANGE to OVERTAKE
            # lane_state_publish.publish(lane_state)
            # actuate(const.BRAKE_SPEED, current_bearing, steer_output)
            # time.sleep(1)

            # elif lane_state == const.CHANGE_LANE:
            #     print("Entering lane change sub-sub-segment")
            #     _dist, _lat, _lon = overtake_lat_lon.split(",")
            #     overtake_location = [float(_lat), float(_lon)]
            #     if not util.has_reached(current_location, overtake_location):
            #         lane_state = const.CHANGE_LANE
            #         lane_state_publish.publish(lane_state)
            #         steer_output, bearing_diff = util.calculate_steer_output_change_lane(current_location, overtake_location, current_bearing, heading)
            #         steer_output = steer_output * -1.00
            #         actuate(const.CHANGE_SPEED, current_bearing, steer_output)
            #     else:steer_output
            #         lane_state = const.OVERTAKE_LANE
            #         lane_state_publish.publish(lane_state)
            # else:
            #     print(f"$$$$ Lane State = {lane_state} $$$$")      
    # OVERTAKE SEGMENT
    # elif lane_state == const.OVERTAKE_LANE:
    #     print("Entering overtake lane segment")
    #     _next_lat, _next_lon = util.get_next_overtake_waypoint(current_location[0], current_location[1])
    #     if collision_warning == const.URGENT_WARNING:
    #         actuate(const.BRAKE_SPEED, current_bearing, steer_output)
    #         # time.sleep(1)
    #     else:
    #         next_location = [_next_lat, _next_lon]
    #         steer_output, bearing_diff = util.calculate_steer_output_change_lane(current_location, next_location, current_bearing, heading)
    #         steer_output = steer_output * -1.00
    #         actuate(const.OVERTAKE_SPEED, current_bearing, steer_output) 
            # time.sleep(1)
    # else:
    #     print(f"$$$$ Lane State = {lane_state} $$$$")
    # NORMAL SEGMENT
    elif lane_state == const.DRIVING_LANE:
        print("Entering normal driving segment")
        #print(f"Lane State = {const.STATE_DICT[lane_state]}")
        steer_output, bearing_diff = calculate_steer_output(current_location, current_bearing)
        
        steer_output = steer_output * -1.00
        const_speed = util.get_speed(collision_warning, lane_state, bearing_diff)
        actuate(const_speed, current_bearing, steer_output)
        if get_distance_to_next_waypoint(current_location) < const.WP_DIST:
            wp = wp + 1

if __name__ == '__main__':    
    global counter, wp, some_var, overtake_location
    
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
            time.sleep(0.1)
            navigation_output(latitude, longitude, current_bearing)
            lane_state_publish.publish(lane_state)
            #print(f"Current waypoint # = {wp}")
            #print(f"Ego vehicle velocity = {float(current_vel)} kmph")  
            print("#####################################\n\n")   
            # with open("driving_waypoints.txt", "a") as file:
            #     for line in navigation_path:
            #         file.write(f"{line}\n")     
        except Exception as e:
            print(f"Inside rospy.is_shutdown() loop: {str(e)}")
            pass
        
        
