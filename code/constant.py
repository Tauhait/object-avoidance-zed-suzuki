from collections import namedtuple

# SPEED VALUES
BRAKE_SPEED = 0
TOP_SPEED = 12
# SPEED VALUES

# MABX CONFIG
MABX_IP = "192.168.50.1"  # Mabx IP for sending from Pegasus
MABX_PORT = 30000  # Mabx port for sending from Pegasus
BUFFER_SIZE = 4096  # packet size
LOCAL_INTERFACE = 'eno1'
NAVIGATION_DATA = 'data.csv'
WAYPOINT_FILENAME = '/usr/local/zed/samples/object-avoidance-zed-suzuki/waypoints_2023-10-06-12:54:39.txt'
# MABX CONFIG

# NAVIGATION
STEER_GAIN = 1200          # For Tight Turns 1200 can be used
TURN_BEARNG_THRESHOLD = 6
# Conversion factor for latitude and longitude to meters
LAT_LNG_TO_METER = 1.111395e5
WP_DIST = 3
RAD_TO_DEG_CONVERSION = 57.2957795
# NAVIGATION

# COLLISION WARNING
MAX_CLASS_ID = 7
MAX_DEPTH = 20
NUM_INTERPOLATED_POINTS = 500
CLASSES = ['person', 'bicycle', 'car', 'motocycle', 'route board', 
           'bus', 'commercial vehicle', 'truck', 'traffic sign', 'traffic light',
            'autorickshaw','stop sign', 'ambulance', 'bench', 'construction vehicle',
            'animal', 'unmarked speed bump', 'marked speed bump', 'pothole', 'police vehicle',
            'tractor', 'pushcart', 'temporary traffic barrier', 'rumblestrips', 'traffic cone', 'pedestrian crossing']
REQ_CLASSES = [0,1,2,3,4,6,7,8,9,10,11,12,13,15,16,17,18,19,20,26]
PERSONS_VEHICLES_CLASSES = [0,1,2,3,4,6,7,8,11,13,15,20,21]
LANE_SPACE = 5
NUM_INTERPOLATED_POINTS = 500
LEFT_RIGHT_DISTANCE = 1.6		#in meteres in either side
STOP_DISTANCE = 10		#in meteres in front of car     #actual 5.5
DETECTING_DISTANCE = 20     #
CAUTION_DISTANCE = 15 
CLOSENESS_THRES = 99999
# COLLISION WARNING

# DRIVESPACE
LABEL = namedtuple( "LABEL", [ "name", "train_id", "color"])
DRIVABLES = [ 
             LABEL("direct", 0, (0, 255, 0)),        # green
             LABEL("alternative", 1, (0, 0, 255)),   # blue
             LABEL("background", 2, (0, 0, 0)),      # black          
            ]
# DRIVESPACE

# OVERTAKE PATH CONFIG
TARGET_REACH = 1
BEARING_ZERO = 0
OVERTAKE_WAYPOINT_DIST = 2
WAIT_TIME = 2000
# OVERTAKE PATH CONFIG

# LANE VELOCITY
DRIVE_SPEED = 8
CHANGE_SPEED = 4
OVERTAKE_SPEED = 6
# LANE VELOCITY

## STATES

# COLLISION WARNING STATES
NO_WARNING = 100
MID_WARNING = 101
URGENT_WARNING = 102
# COLLISION WARNING STATES

# COLLISION AVOIDANCE
OVERTAKE = 200
CONTINUE = 201
SWITCH = 202
# COLLISION AVOIDANCE

# LANE STATES
DRIVING_LANE = 300
CHANGE_LANE = 301
OVERTAKE_LANE = 302
# LANE STATES

# OPTICAL FLOW
TRAFFIC_FROM_LEFT = 400
SAFE_TO_OVERTAKE = 401
TRAFFIC_FROM_RIGHT = 402
# OPTICAL FLOW

STATE_DICT = {
    100: "NO_WARNING",
    101: "MID_WARNING",
    102: "URGENT_WARNING",
    200: "OVERTAKE",
    201: "CONTINUE",
    202: "SWITCH",
    300: "DRIVING_LANE",
    301: "CHANGE_LANE",
    302: "OVERTAKE_LANE",
    400: "TRAFFIC_FROM_LEFT",
    401: "SAFE_TO_OVERTAKE",
    402: "TRAFFIC_FROM_RIGHT",
}

DECISION_THRESHOLD = 15




