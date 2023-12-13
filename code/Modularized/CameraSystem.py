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

class CameraSystem:

    def __init__(self, device):
        self.device = device
        self.capture_thread = None
        self.zed = None
        self.init_params = sl.InitParameters()
        self.runtime_params = sl.RuntimeParameters()
        self.viewer = None
        self.point_cloud_res = sl.Resolution(720, 404)
        self.point_cloud = sl.Mat(self.point_cloud_res.width, self.point_cloud_res.height, sl.MAT_TYPE.F32_C4, sl.MEM.CPU)
        self.image_left = sl.Mat()
        self.camera_res = None
        self.display_resolution = sl.Resolution(1280, 720)
        self.image_scale = [self.display_resolution.width / self.camera_res.width, self.display_resolution.height / self.camera_res.height]
        self.image_left_ocv = np.full((self.display_resolution.height, self.display_resolution.width, 4), [245, 239, 239, 255], np.uint8)
        self.tracks_resolution = sl.Resolution(400, self.display_resolution.height)
        self.track_view_generator = None
        self.image_track_ocv = np.zeros((self.tracks_resolution.height, self.tracks_resolution.width, 4), np.uint8)
        self.cam_w_pose = sl.Pose()
        self.depth_map = sl.Mat()
        self.lane_state = const.DRIVING_LANE
        self.exit_signal = False

    def run(self):
        # Start capture thread
        self.start_capture_thread()

        # Initialize camera
        self.initialize_camera()

        # Start OpenGL viewer
        self.init_viewer()

        while not self.exit_signal:
            if self.zed.grab(self.runtime_params) == sl.ERROR_CODE.SUCCESS:
                # Retrieve image and wait for detections
                self.retrieve_image_and_detections()

                # Retrieve objects, depth map, and publish lane state
                self.retrieve_objects_and_depth()
                self.publish_lane_state()

                # Process and display data
                self.process_and_display_data()
            else:
                self.exit_signal = True

        # Stop capture thread
        self.stop_capture_thread()

        # Close camera and viewer
        self.close_camera_and_viewer()

    def start_capture_thread(self):
        # TODO: implement capture thread logic

    def stop_capture_thread(self):
        # TODO: implement capture thread stopping logic

    def initialize_camera(self):
        # Set input type based on presence of svo file
        self.init_params.input_type.set_from_camera_id(1)
        if opt.svo is not None:
            self.init_params.input_type.set_from_svo_file(opt.svo)

        # Configure camera parameters
        self.init_params.camera_resolution = sl.RESOLUTION.HD720
        self.init_params.coordinate_units = sl.UNIT.METER
        self.init_params.depth_mode = sl.DEPTH_MODE.ULTRA
        self.init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
        self.init_params.depth_maximum_distance = 50

        # Open camera
        self.zed = sl.Camera()
        status = self.zed.open(self.init_params)
        if status != sl.ERROR_CODE.SUCCESS:
            print(repr(status))
            exit()

        # Retrieve camera information
        camera_infos = self.zed.get_camera_information()
        self.camera_res = camera_infos.camera_configuration.resolution

    def init_viewer(self):
        self.viewer = gl.GLViewer()
        self.viewer.init(camera_infos.camera_model, self.point_cloud_res, True)

    def retrieve_image_and_detections(self):
        # Retrieve left image
        image_left_tmp = sl.Mat()
        self.zed.retrieve_image(image_left_tmp, sl.VIEW.LEFT)
        self.image_net = image_left_tmp.get_data()

        # Set flag for detection thread
        run_signal = True

        # Wait for detections
        while run_signal:
            sleep(0.001)

        # Lock and ingest detections
        lock.acquire()
        self.zed.ingest_custom_box_objects(detections)
        lock.release()

    def retrieve_objects_and_depth(self):
        # Retrieve objects and depth map
        objects = sl.Objects()
        obj_runtime_param = sl.ObjectDetectionRuntimeParameters()
        self.zed.retrieve_objects(objects, obj_runtime_param)
        self.zed.retrieve_measure(self.depth_map, sl.MEASURE.DEPTH, sl.MEM.CPU)

    def publish_lane_state(self):
        # Publish lane state based on detected objects
        # TODO: implement lane state detection and publishing logic

    def process_and_display_data(self):
        # Retrieve point cloud and copy to renderable format
        self.zed.retrieve_measure(self.point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU, self.point_cloud_res)
        self.point_cloud.copy_to(point_cloud_render)

        # Retrieve left image and scale for display
        self.zed.retrieve_image(self.image_left, sl.VIEW.LEFT, sl.MEM.CPU, self.display_resolution)
        np.copyto(self.image_left_ocv, self.image_left.get_data())

        # Update OpenGL viewer with point cloud and objects
        self.viewer.updateData(point_cloud_render, objects)

        # Render 2D content on left image
        cv_viewer.render_2D(self.image_left_ocv, self.image_scale, objects, True)

        # Generate and display tracks view
        self.zed.get_position(self.cam_w_pose, sl.REFERENCE_FRAME.WORLD)
        self.track_view_generator.generate_view(objects, self.cam_w_pose, self.image_track_ocv, objects.is_tracked)

        # Display collision warning and segmentation images
        cv2.imshow("Collision Warning", self.image_net)
        cv2.imshow("Segmentation", overlay_image)

        # Handle key presses
        key = cv2.waitKey(10)
        if key == 27:
            self.exit_signal = True

    def close_camera_and_viewer(self):
        # Close camera
        self.zed.close()

        # Close viewer
        self.viewer.exit()

        

