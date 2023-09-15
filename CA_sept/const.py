from collections import namedtuple

DRIVING_LANE = 0
CHANGE_LANE = 1
OVERTAKE_LANE = 2

TOP_SPEED = 12

MABX_IP = "192.168.50.1"  # Mabx IP for sending from Pegasus
MABX_PORT = 30000  # Mabx port for sending from Pegasus
BUFFER_SIZE = 4096  # packet size
LOCAL_INTERFACE = 'eno1'
NAVIGATION_DATA = 'data.csv'
WAYPOINT_FILENAME = '/usr/local/zed/samples/object-avoidance-zed-suzuki/waypoints_2023-09-08-11:36:13.txt'
STEER_GAIN = 1200          # For Tight Turns 1200 can be used
TURN_BEARNG_THRESHOLD = 6
# Conversion factor for latitude and longitude to meters
LAT_LNG_TO_METER = 1.111395e5
WP_DIST = 3
RAD_TO_DEG_CONVERSION = 57.2957795

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

LABEL = namedtuple( "LABEL", [ "name", "train_id", "color"])
DRIVABLES = [ 
             LABEL("direct", 0, (0, 255, 0)),        # green
             LABEL("alternative", 1, (0, 0, 255)),   # blue
             LABEL("background", 2, (0, 0, 0)),      # black          
            ]

CLOSENESS_THRES = 99999

LEFT_SIDE = -1
SAFE_TO_OVERTAKE = 0
RIGHT_SIDE = 1

OVERTAKE = 1
CONTINUE = 0
SWITCH = 2

TARGET_REACH = 1

BEARING_ZERO = 0
OVERTAKE_WAYPOINT_DIST = 2