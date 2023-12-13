import util
import CollisionAvoidance
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

class Perception:

    def __init__(self):
        
        rospy.init_node('perception_front_cam', anonymous=True)

        self.lock = Lock()
        self.detections = []
        self.image_net = None
        self.depth_map = None

        self.model = YOLO(opt.weights)
        self.model.to(device)

        # Initialize camera
        self.zed = sl.Camera()

        # Set configuration parameters
        self.init_params = sl.InitParameters()

        if opt.svo is not None:
            self.init_params.input_type.set_from_svo_file(opt.svo)

        # Set camera resolution and depth mode
        self.init_params.camera_resolution = sl.RESOLUTION.HD720
        self.init_params.coordinate_units = sl.UNIT.METER
        self.init_params.depth_mode = sl.DEPTH_MODE.ULTRA
        self.init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
        self.init_params.depth_maximum_distance = 50

        self.runtime_params = sl.RuntimeParameters()

        # Open the camera
        self.open_camera()

        # Enable positional tracking and object detection
        self.enable_positional_tracking()
        self.enable_object_detection()

        self.objects = sl.Objects()
        self.obj_runtime_param = sl.ObjectDetectionRuntimeParameters()

        # Initialize OpenGL viewer
        self.viewer = gl.GLViewer()

        # Set point cloud resolution
        point_cloud_res = sl.Resolution(min(camera_res.width, 720), min(camera_res.height, 404))
        self.point_cloud = sl.Mat(point_cloud_res.width, point_cloud_res.height, sl.MAT_TYPE.F32_C4, sl.MEM.CPU)
        self.point_cloud_render = sl.Mat()

        # Initialize display resolution and image buffers
        display_resolution = sl.Resolution(min(camera_res.width, 1280), min(camera_res.height, 720))
        self.image_scale = [display_resolution.width / camera_res.width, display_resolution.height / camera_res.height]
        self.image_left_ocv = np.full((display_resolution.height, display_resolution.width, 4), [245, 239, 239, 255], np.uint8)

        # Initialize tracks view generator
        camera_config = camera_infos.camera_configuration
        tracks_resolution = sl.Resolution(400, display_resolution.height)
        self.track_view_generator = cv_viewer.TrackingViewer(tracks_resolution, camera_config.fps, init_params.depth_maximum_distance)
        self.track_view_generator.set_camera_calibration(camera_config.calibration_parameters)
        self.image_track_ocv = np.zeros((tracks_resolution.height, tracks_resolution.width, 4), np.uint8)

        # Initialize camera pose
        self.cam_w_pose = sl.Pose()

        # Start collision avoidance thread
        self.collision_avoidance = CollisionAvoidance()
        self.collision_avoidance.start_collision_avoidance()

    def open_camera(self):
        status = self.zed.open(self.init_params)
        if status != sl.ERROR_CODE.SUCCESS:
            print(repr(status))
            exit()

    def enable_positional_tracking(self):
        positional_tracking_parameters = sl.PositionalTrackingParameters()

        # Uncomment the following line if the camera is static
        # positional_tracking_parameters.set_as_static = True

        self.zed.enable_positional_tracking(positional_tracking_parameters)

    def enable_object_detection(self):
        obj_param = sl.ObjectDetectionParameters()
        obj_param.detection_model = sl.OBJECT_DETECTION_MODEL.CUSTOM_BOX_OBJECTS
        obj_param.enable_tracking = True
        self.zed.enable_object_detection(obj_param)

    def main_loop(self):
        while not self.collision_avoidance.exit_signal:
            if not self.zed.grab(self.runtime_params):
                break

            self.zed.retrieve_objects(self.objects, self.obj_runtime_param)
            self.zed.retrieve_measure(self.measure, sl.MEASURE.TRACKING_POSE)

            self.cam_w_pose.set_position(sl.Translation(self.measure.pose.tx, self.measure.pose.ty, self.measure.pose.tz))
            self.cam_w_pose.set_rotation(sl.Orientation(self.measure.pose.qw, self.measure.pose.qx, self.measure.pose.qy, self.measure.pose.qz))

            # Update point cloud
            self.zed.retrieve_measure(self.measure, sl.MEASURE.DEPTH)
            self.point_cloud.set(self.measure.depth_data)
            self.viewer.update_point_cloud(self.point_cloud, self.point_cloud_render)

            # Update object detection results
            with self.lock:
                self.detections = self.process_object_detection(self.objects)

            # Update image and tracks view
            self.zed.retrieve_image(self.left_image)
            self.merge_images()
            self.track_view_generator.update_tracks(self.objects, self.cam_w_pose)
            self.image_track_ocv = cv2.copyMakeBorder(self.track_view_generator.get_image(), 0, 0, 0, display_resolution.width - tracks_resolution.width, cv2.BORDER_CONSTANT, value=(0, 0, 0))

            # Update viewer
            if self.viewer.is_visible:
                self.viewer.update(self.image_left_ocv, self.image_track_ocv, self.detections)
                self.viewer.update_camera_pose(self.cam_w_pose)

            # Update collision avoidance with detections
            self.collision_avoidance.detections = self.detections

            cv2.imshow("ZED", self.image_left_ocv)
            cv2.waitKey(1)

        self.collision_avoidance.stop_collision_avoidance()
        self.zed.close()

