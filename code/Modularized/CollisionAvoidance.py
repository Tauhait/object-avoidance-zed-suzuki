#!/usr/bin/env python3
import ActuationController
import TrajectoryGenerator
import OpticalFlowPerception
import util
import constant as const
import sys
import numpy as np
import time
import argparse
import torch
import cv2
import pyzed.sl as sl
import warnings, random
import math
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import ogl_viewer.viewer as gl
import cv_viewer.tracking_viewer as cv_viewer
import rospy
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from std_msgs.msg import Float32, String, Int8, Int32
from ultralytics import YOLO
from time import sleep
from math import asin, atan2, cos, degrees, radians, sin, sqrt
from torchvision import transforms
from collections import namedtuple
from novatel_oem7_msgs.msg import BESTPOS, BESTVEL, INSPVA
from seg_model.pspnet import PSPNet
from scipy.interpolate import CubicSpline
from threading import Lock, Thread
warnings.filterwarnings("ignore")


class CollisionAvoidance:

    def __init__(self):
        # ROS publishers
        self.collision_warning_publish = rospy.Publisher('collision_warning', Int32, queue_size=1)
        self.collision_avoidance_publish = rospy.Publisher('collision_avoidance', Int32, queue_size=1)
        self.overtake_lat_lon_publish = rospy.Publisher('overtake_lat_lon', String, queue_size=1)
        self.lane_state_publish = rospy.Publisher('lane_state', Int32, queue_size=1)

        # Class attributes for communication
        self.lat = None
        self.lon = None
        self.lane_state = None

        # Red and green pixel lists for object detection
        self.red_pixels = []
        self.green_pixels = []

        # Flags and variables for control
        self.show_plot = True
        self.run_signal = False
        self.exit_signal = False

        # Define object references
        
        self.perception = Perception()
        self.actuation_controller = ActuationController()
        self.trajectory_generator = TrajectoryGenerator(...)  # provide initial and target poses
        self.optical_flow_perception = OpticalFlowPerception()

        # Train ID to color mapping
        self.train_id_to_color = np.array([c.color for c in const.DRIVABLES if (c.train_id != -1 and c.train_id != 255)])

        # Subscribe to lane state topic
        rospy.Subscriber("/lane_state", Int8, self.callback_lane_state)

        # Subscribe to bestpos topic
        rospy.Subscriber("/novatel/oem7/bestpos", BESTPOS, self.callback_latlong)

    def callback_lane_state(self, data):
        self.lane_state = data.data

    def callback_latlong(self, data):
        self.lat = data.lat
        self.lon = data.lon

    def start_collision_avoidance(self):
        # Thread to run collision avoidance logic
        thread = Thread(target=self.collision_avoidance_loop)
        thread.daemon = True
        thread.start()