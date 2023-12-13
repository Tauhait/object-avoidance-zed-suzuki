import random, numpy as np, math
import constant as const
import pyzed.sl as sl
from scipy.interpolate import CubicSpline
from math import asin, atan2, cos, degrees, radians, sin, sqrt

class ActuationController:
    def set_angle(self, m_nAngle, deltaAngle):
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

    def calculate_steer_output_change_lane(self, currentLocation, targetLocation, current_bearing, heading):
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

    def calc_checksum(self, msg):
        cs = 0
        for m in msg:
            cs += m
        cs = (0x00 - cs) & 0x000000FF
        cs = cs & 0xFF
        return cs

    def set_speed(self, speed):
        speed = speed * 128
        H_speed = (int)(speed) >> 8
        L_speed = (int)(speed) & 0xff
        return H_speed, L_speed

    def get_msg_to_mabx(self, speed, m_nAngle, angle, flasher, counter):
        H_Angle, L_Angle = self.set_angle(m_nAngle, -1*angle)
        H_Speed, L_Speed = self.set_speed(speed)

        msg_list = [1, counter, 0, 1, 52, 136, 215, 1, H_Speed, L_Speed, H_Angle, L_Angle, 0, flasher, 0, 0, 0, 0]

        msg_list[2] = self.calc_checksum(msg_list)
        message = bytearray(msg_list)
        print("Speed: ", message[8], message[9])
        print("Angle: ", message[10], message[11])
        print("===============================================================================")
        return message

    def get_flasher(self, angle):
        return 1 if angle > 90 else 2 if angle < -100 else 0
    
    def get_bearing(self, hyp, base=1.6):
        bearing = 0
        if hyp >= base:
            perp = math.sqrt(hyp**2 - base**2)
            bearing = math.atan(perp / base)
        return 90 - math.degrees(bearing)
    
    def get_speed(self, collision_warning, lane_state, bearing_diff):
        # print("Inside get_speed")
        const_speed = const.DRIVE_SPEED
        if lane_state == const.DRIVING_LANE:
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
        else:
            if collision_warning == const.URGENT_WARNING:
                const_speed = const.BRAKE_SPEED
            else:
                const_speed = const.OVERTAKE_SPEED

        return const_speed
    
    def has_reached(self, current_loc, target_loc):
        distance_to_target = np.linalg.norm(np.array(current_loc) - target_loc) * const.LAT_LNG_TO_METER
        return distance_to_target < const.TARGET_REACH