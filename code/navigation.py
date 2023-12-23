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


rospy.init_node('navigation', anonymous=True)
lane_state_publish = rospy.Publisher('lane_state', Int32, queue_size=1)

# call back functions
def callback_zed_camera_imu(data):
    global acc_z
    acc_z = 0 
    acc_z = data.linear_acceleration.z

def callback_lane_state(data):
    global lane_state
    lane_state = data.data
    
def callback_collision_warning(data):
    global cw_warning
    cw_warning = const.NO_WARNING
    cw_warning = data.data
    
def callback_collision_avoidance(data):
    global collision_avoidance
    collision_avoidance = data.data
    
def callback_ob_detect_class(data):
    global ob_detect_class
    ob_detect_class = data.data
    
def callback_optical_flow(data):
    global optical_flow
    optical_flow = data.data
   
def callback_vel(data):
    global current_vel 
    current_vel = 3.6 * data.hor_speed  

def callback_heading(data):
    global heading
    heading = data.azimuth  

def callback_latlong(data):
    global lat, lng, lat_delta, lng_delta
    lat = data.lat
    lng = data.lon
    lat_delta = data.lat_stdev
    lng_delta = data.lon_stdev

# ros node subscribes to
rospy.Subscriber("/lane_state", Int32, callback_lane_state)
rospy.Subscriber("/collision_warning", Int32, callback_collision_warning)
rospy.Subscriber("/collision_avoidance", Int32, callback_collision_avoidance)
rospy.Subscriber("/ob_detect_class", Int32, callback_ob_detect_class)
rospy.Subscriber("/optical_flow", Int32, callback_optical_flow)
rospy.Subscriber("/novatel/oem7/bestvel",BESTVEL, callback_vel)
rospy.Subscriber("/novatel/oem7/inspva",INSPVA, callback_heading)
rospy.Subscriber("/novatel/oem7/bestpos",BESTPOS, callback_latlong)


def actuate(speed, steer_output, indicator):
    global counter
    flasher = indicator
    counter = (counter + 1) % 256
    message = util.get_msg_to_mabx(speed, steer_output, 0, flasher, counter)
    MABX_SOCKET.sendto(message, MABX_ADDR)

def log_and_print(str):
    logger.info(str)
    print(str)

def navigation_output(latitude, longitude, current_bearing, cw_warning):
    global counter, speed, wp, lane_state, collision_avoidance, ob_detect_class, optical_flow
    global driving_ind, current_vel, overtake_wp, overtake_coords_list, num_waypoints, obstacle_to_overtake
    flasher = const.BOTH_INDICATOR    #For indicator: 0 None, 1 Left, 2 Right, 3 Both;
    counter = (counter + 1) % 256

    const_speed = const.TOP_SPEED
    current_location = [latitude, longitude]
    distance_to_final_waypoint = np.linalg.norm(np.array(current_location) - WAYPOINTS[-1]) * const.LAT_LNG_TO_METER


    if driving_ind:
        lane_state_publish.publish(const.DRIVING_LANE)
        if (distance_to_final_waypoint > 1 and wp < WP_LEN):     # to check if the final point is not less than 1m
            if cw_warning == const.MID_WARNING:
                log_and_print(f"Collision Warning Status : Caution")
            elif cw_warning == const.URGENT_WARNING:
                log_and_print(f"Collision Warning Status : Brake Signal") 
            else:
                log_and_print(f"Collision Warning Status : Safe")
            
            steer_output, bearing_diff = util.calculate_steer_output(current_location, current_bearing, heading, wp, WAYPOINTS)
            steer_output *= -1.0

            future_bearing_diff = util.calculate_bearing_difference_for_speed_reduction(current_location, current_bearing, heading, WAYPOINTS, wp, WP_LEN)

            if wp < 1 or wp > WP_LEN:
                const_speed = const.TURNING_FACTOR * const.TOP_SPEED
            
            if abs(bearing_diff) > const.BEARING_DIFF_THRESHOLD:
                const_speed = const.TURNING_FACTOR * const.TOP_SPEED
                log_and_print(f"Turning Speed from code : {const_speed:.0f} kmph")
            elif abs(future_bearing_diff) > const.BEARING_DIFF_THRESHOLD:
                const_speed = const.SPEED_REDUCTION_FACTOR * const_speed
                log_and_print(f"Curve Speed from code : {const_speed:.0f} kmph")

            if cw_warning == const.MID_WARNING:
                const_speed = const.TURNING_FACTOR * const.TOP_SPEED
            elif cw_warning == const.URGENT_WARNING:
                const_speed = const.BRAKE_SPEED
                # editing usage of lane_state, collision_avoidance, ob_detect_class, optical_flow
                log_and_print(f"Collision Warning = {const.STATE_DICT[cw_warning]},\n")
                log_and_print(f"Collision Avoidance = {const.STATE_DICT[collision_avoidance]},\n")
                log_and_print(f"Lane State = {const.STATE_DICT[lane_state]},\n")
                log_and_print(f"Optical Flow = {const.STATE_DICT[optical_flow]}\n")
                if int(current_vel) == 0:
                    if ob_detect_class == const.OBJ_CLASS_CAR or const.OBJ_CLASS_CYCLE:
                        const.set_dynamic_obstacle(WAYPOINTS[wp][0], WAYPOINTS[wp][1])
                    else:
                        const.set_dynamic_obstacle(WAYPOINTS[wp + 2][0], WAYPOINTS[wp + 2][1])                   
                    driving_ind = False
                    overtake_wp = 0
                    log_and_print(f"\n\nset_dynamic_obstacle is set\n\n")
                    log_and_print(f"cw_warning = {const.STATE_DICT[cw_warning]}")
                    num_waypoints = 0
                    log_and_print(f"ob_detect_class = {str(const.CLASSES[ob_detect_class])}")
                    if ob_detect_class == const.OBJ_CLASS_CAR:
                        num_waypoints = len(const.DISTANCE_CAR)
                    elif ob_detect_class == const.OBJ_CLASS_CYCLE:
                        num_waypoints = len(const.DISTANCE_CYCLE)
                    else:
                        num_waypoints = len(const.DISTANCE_PED)
                    
                    obstacle_to_overtake = ob_detect_class
            
            distance_to_nextpoint = np.linalg.norm(np.array(current_location) - WAYPOINTS[wp]) * const.LAT_LNG_TO_METER
            log_and_print(f"{wp} out of {WP_LEN} | Next Coordinate distance : {distance_to_nextpoint:.1f} m")
            try:
                if wp < WP_LEN and distance_to_nextpoint < const.LOOK_AHEAD_DISTANCE:
                    wp += 1
            except IndexError:
                log_and_print("Waypoint index out of range! - Seems like you are at wrong location or Inputted wrong waypoint")
        else:
            log_and_print("----- FINISHED  -----")
            log_and_print("Brake Activated")
            steer_output = 0
            const_speed = 0
            flasher = 0
    else:
        lane_state_publish.publish(const.CHANGE_LANE)
        if num_waypoints is None:
            raise Exception("num_waypoints for overtaking is null")
        if obstacle_to_overtake is None:
            raise Exception("obstacle object class for overtaking is null")
        if overtake_wp < num_waypoints:
            if overtake_wp == 0:
                util.gen_trail_of_waypoints(const.OBS_LAT, const.OBS_LON, obstacle_to_overtake)
                print("Trail of waypoints generated for obstacle avoidance")
                overtake_coords_list = util.get_coordinates(const.FILENAME_TRAIL)
                print(f"len of overtake_coords_list = {len(overtake_coords_list)}")
            
            if overtake_coords_list is not None:
                steer_output, bearing_diff = util.calculate_steer_output(current_location, current_bearing, heading, overtake_wp, overtake_coords_list)
                steer_output *= -1.0
                # if cw_warning == const.URGENT_WARNING:
                #     const_speed = const.BRAKE_SPEED
                # else:
                #     const_speed = const.OVERTAKE_SPEED
                const_speed = const.OVERTAKE_SPEED
                
                distance_to_next_overtake_point = np.linalg.norm(np.array(current_location) - overtake_coords_list[overtake_wp]) * const.LAT_LNG_TO_METER
                log_and_print(f"{overtake_wp} out of {len(overtake_coords_list)} | Next overtake coordinate distance : {distance_to_next_overtake_point:.1f} m")
                log_and_print(f"Collision Avoidance = {const.STATE_DICT[collision_avoidance]},\n")
                # if optical_flow == const.TRAFFIC_FROM_RIGHT:
                #     const_speed = const.BRAKE_SPEED
                
                try:
                    if overtake_wp < num_waypoints and distance_to_next_overtake_point < const.OVERTAKE_LOOK_AHEAD_DISTANCE:
                        overtake_wp += 1
                except IndexError:
                    log_and_print("Waypoint index out of range! - Seems like you are at wrong location or Inputted wrong waypoint")
            else:
                log_and_print("----- FINISHED  -----")
                log_and_print("Brake Activated")
                steer_output = 0
                const_speed = 0
                flasher = 0
                
        else:
            driving_ind = True
            log_and_print(f"Inside else part driving_ind = {driving_ind}")
            steer_output = 0
            const_speed = const.BRAKE_SPEED
            flasher = 0
            obstacle_to_overtake = None
            num_waypoints = None

        distance_to_nextpoint = np.linalg.norm(np.array(current_location) - WAYPOINTS[wp]) * const.LAT_LNG_TO_METER
        log_and_print(f"{wp} out of {WP_LEN} | Next original path coordinate distance : {distance_to_nextpoint:.1f} m")           
        try:
            if wp < WP_LEN and distance_to_nextpoint < const.LOOK_AHEAD_DISTANCE_DURING_OVERTAKE:
                wp += 1
        except IndexError:
            log_and_print("Waypoint index out of range! - Seems like you are at wrong location or Inputted wrong waypoint")
            
    log_and_print(f"Lane State from navigation = {const.STATE_DICT[lane_state]}")
    log_and_print(f"Speed set by code: {const_speed:.0f} kmph")
    log_and_print(f"heading = {heading}")
    try:
        actuate(const_speed, steer_output, flasher)
    except Exception as e:
        log_and_print(f"Error sending message to MABX: {e}")

def main_loop():
    global num_waypoints, obstacle_to_overtake
    num_waypoints = None
    obstacle_to_overtake = None
    while not rospy.is_shutdown():
        try:
            log_and_print(f"Current Coordinate No. : {wp}")
            log_and_print(" ")
            log_and_print(f"Velocity in kmph as per GNSS = {current_vel:.0f} kmph")
            
            latitude = float(lat)
            longitude = float(lng)
            current_bearing = float(heading)
            navigation_output(latitude, longitude, current_bearing, cw_warning)
            time.sleep(const.SLEEP_INTERVAL/1000)           
        except KeyboardInterrupt:       #Currently not working
            log_and_print("Autonomous Mode is terminated manually!")
            actuate(const.BRAKE_SPEED, const.ZERO_STEET_OUTPUT, const.NO_INDICATOR)
            raise SystemExit
        
        except Exception as e:
            log_and_print(f"An error occurred: {e}")


if __name__ == '__main__':    
    global speed, reduction_factor, steer_output, counter, wp, driving_ind, overtake_wp, overtake_coords_list, cw_warning
    
    log_dir = "devLogs"
    logger = util.setup_logging(log_dir, const.WAYPOINT_FILENAME)
    logger.info('Development Code Starting')

    speed = const.TOP_SPEED
    wp = 0
    steer_output = 0
    counter = 0
    driving_ind = True
    overtake_wp = 0
    overtake_coords_list = []
    main_loop()
        
        
