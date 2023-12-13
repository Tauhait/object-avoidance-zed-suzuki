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

class AutonomousDrivingController:

    @classmethod
    def wp(cls):
        return cls._wp

    @classmethod
    def set_wp(cls, value):
        cls._wp = value

    @classmethod
    def counter(cls):
        return cls._counter

    @classmethod
    def set_counter(cls, value):
        cls._counter = value
    
    def __init__(self):
        self.MABX_SOCKET = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.MABX_ADDR = (const.MABX_IP, const.MABX_PORT)
        self.WAYPOINTS = util.get_coordinates(const.WAYPOINT_FILENAME)
        self.WP_LEN = len(self.WAYPOINTS)

        self.lat = None
        self.lng = None
        self.collision_avoidance = None
        self.optical_flow = None
        self.overtake_lat_lon = None
        self.lane_state = None
        self.collision_warning = None

        rospy.init_node('navigation', anonymous=True)
        self.lane_state_publish = rospy.Publisher('lane_state', Int32, queue_size=1)

        self.subscribe_topics()
        # Initialize class variables
        self._counter = 0
        self._wp = 0
    
    def subscribe_topics(self):
        rospy.Subscriber("/lane_state", Int32, self.callback_lane_state)
        rospy.Subscriber("/collision_warning", Int32, self.callback_collision_warning)
        rospy.Subscriber("/collision_avoidance", Int32, self.callback_collision_avoidance)
        rospy.Subscriber("/overtake_lat_lon", String, self.callback_streering_angle)
        rospy.Subscriber("/optical_flow", Int32, self.callback_optical_flow)
        rospy.Subscriber("/novatel/oem7/bestvel", BESTVEL, self.callback_vel)
        rospy.Subscriber("/novatel/oem7/inspva", INSPVA, self.callback_heading)
        rospy.Subscriber("/novatel/oem7/bestpos", BESTPOS, self.callback_latlong)
    

    def callback_lane_state(self, data):
        self.lane_state = data.data

    def callback_collision_warning(self, data):
        self.collision_warning = data.data

    def callback_collision_avoidance(self, data):
        self.collision_avoidance = data.data

    def callback_streering_angle(self, data):
        self.overtake_lat_lon = data.data

    def callback_optical_flow(self, data):
        self.optical_flow = data.data

    def callback_vel(self, data):
        self.current_vel = 3.6 * data.hor_speed

    def callback_heading(self, data):
        self.heading = data.azimuth

    def callback_latlong(self, data):
        self.lat = data.lat
        self.lng = data.lon
    
    def actuate(self, const_speed, current_bearing, steer_output):
        flasher = util.get_flasher(current_bearing)
        self._counter = (self._counter + 1) % 256
        message = util.get_msg_to_mabx(const_speed, steer_output, 0, flasher, self._counter)
        self.MABX_SOCKET.sendto(message, self.MABX_ADDR)
        print(f"Actuation command send to MABX = {str(message)}")

    def get_distance_to_next_waypoint(self, current_location):
        distance_to_nextpoint = np.linalg.norm(np.array(current_location) - self.WAYPOINTS[self._wp]) * const.LAT_LNG_TO_METER
        print(f"Distance between last and current waypoint = {distance_to_nextpoint} m")
        return distance_to_nextpoint

    def has_reached_end(self):
        return self._wp >= self.WP_LEN

    def calculate_steer_output(self, current_location, current_bearing):
        off_y = - current_location[0] + self.WAYPOINTS[self._wp][0]
        off_x = - current_location[1] + self.WAYPOINTS[self._wp][1]

        # calculate bearing based on position error
        target_bearing = 90.00 + math.atan2(-off_y, off_x) * const.RAD_TO_DEG_CONVERSION

        # convert negative bearings to positive by adding 360 degrees
        if target_bearing < 0:
            target_bearing += 360.00

        # calculate the difference between heading and bearing
        bearing_diff = current_bearing - target_bearing

        # normalize bearing difference to range between -180 and 180 degrees
        if bearing_diff < -180:
            bearing_diff += 360

        if bearing_diff > 180:
            bearing_diff -= 360

        steer_output = const.STEER_GAIN * np.arctan(-1 * 2 * 3.5 * np.sin(np.radians(bearing_diff)) / 8)

        return steer_output, bearing_diff


    def navigation_output(self, latitude, longitude, current_bearing):
        """
        Determines the navigation output based on the current sensor data and state.

        Args:
            latitude: Current latitude of the vehicle.
            longitude: Current longitude of the vehicle.
            current_bearing: Current heading of the vehicle.
        """

        current_location = [latitude, longitude]

        # Check if destination has been reached
        if self.has_reached_end():
            while True:
                self.actuate(const.BRAKE_SPEED, const.BEARING_ZERO)
                print("Reached destination !!!!!!!!!!!!!!")

        # Update internal state based on sensor data
        self.collision_warning = int(self.collision_warning)

        # Handle collision or lane change segments
        if (
            (int(self.current_vel) == const.BRAKE_SPEED and self.collision_warning == const.URGENT_WARNING)
            or self.lane_state != const.DRIVING_LANE
        ):
            self._collision_or_lane_change_segment(current_location, current_bearing)
        else:
            # Normal driving segment
            self._normal_driving_segment(current_location, current_bearing)
        
    
    def _collision_or_lane_change_segment(self, current_location, current_bearing):
        """
        Handles navigation logic within collision or lane change segments.

        Args:
            current_location: Current location of the vehicle.
            current_bearing: Current heading of the vehicle.
        """

        print("Entering collision or lane change segment")
        print(
            f"collision_warning = {const.STATE_DICT[self.collision_warning]}, "
            f"collision_avoidance = {const.STATE_DICT[self.collision_avoidance]}, "
            f"Lane_state = {const.STATE_DICT[self.lane_state]}, "
            f"BESTVEL = {self.current_vel}, "
            f"HEADING = {self.heading}, "
            f"BESTPOS = {current_location}, "
            f"overtake_lat_lon = {self.overtake_lat_lon}, "
            f"optical_flow = {self.optical_flow}")

        # Handle overtake or lane change
        if self.collision_avoidance == const.OVERTAKE or self.lane_state == const.CHANGE_LANE:
            self._overtake_or_lane_change_sub_segment(current_location, current_bearing)

        # Handle overtake lane
        elif self.lane_state == const.OVERTAKE_LANE:
            self._overtake_lane_sub_segment(current_location, current_bearing)

        else:
            print(f"$$$$ Lane State = {self.lane_state} $$$$")

    def _normal_driving_segment(self, current_location, current_bearing):
        """
        Handles navigation logic within the normal driving segment.

        Args:
            current_location: Current location of the vehicle.
            current_bearing: Current heading of the vehicle.
        """

        print("Entering normal driving segment")
        print(
            f"collision_warning = {const.STATE_DICT[self.collision_warning]}, "
            f"collision_avoidance = {const.STATE_DICT[self.collision_avoidance]}, "
            f"Lane_state = {const.STATE_DICT[self.lane_state]}, "
            f"BESTVEL = {self.current_vel}, "
            f"HEADING = {self.heading}, "
            f"BESTPOS = {current_location}, "
            f"overtake_lat_lon = {self.overtake_lat_lon}, "
            f"optical_flow = {self.optical_flow}")

        # Calculate steering output and speed
        steer_output, bearing_diff = self.calculate_steer_output(current_location, current_bearing)
        steer_output *= -1.00
        const_speed = util.get_speed(self.collision_warning, self.lane_state, bearing_diff)

        # Send actuation commands
        self.actuate(const_speed, current_bearing, steer_output)

        # Check if next waypoint is reached
        if self.get_distance_to_next_waypoint(current_location) < const.WP_DIST:
            self._wp += 1














