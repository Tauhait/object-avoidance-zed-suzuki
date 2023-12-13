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

class CollisionWarning:

    def __init__(self, img):
        # Define object references
        self.image_net = img
        self.collision_warning_publish = rospy.Publisher('collision_warning', Int32, queue_size=1)

    def draw_bbox(self, object, color, depth_map):
        # Access instance variable
        #for object in objects.object_list:
        xA = int(object.bounding_box_2d[0][0])
        yA = int(object.bounding_box_2d[0][1])
        xB = int(object.bounding_box_2d[2][0])
        yB = int(object.bounding_box_2d[2][1])

        c1, c2 = (xA, yA), (xB, yB) 
        h_x = 650
        h_y = 720
        center_point = round((c1[0] + c2[0]) / 2), round((c1[1] + c2[1]) / 2) ## center of object
        angle = util.get_angle_between_horizontal_base_object_center(h_x, center_point[0], h_x, self.image_net.shape[1], 
                                                                h_y, center_point[1], h_y, self.image_net.shape[0])
        #dist = math.sqrt(object.position[0]*object.position[0] + object.position[1]*object.position[1])
        #vel = math.sqrt(object.velocity[0]*object.velocity[0] + object.velocity[1]*object.velocity[1])

        depth = util.get_object_depth_val(center_point[0], center_point[1], depth_map)
        cv2.line(self.image_net, (h_x, h_y), (center_point[0], center_point[1]), color, 1)
        # cv2.line(self.image_net, (self.image_net.shape[1], self.image_net.shape[0]), (center_point[0], center_point[1]), color, 1)
        cv2.rectangle(self.image_net, (xA, yA), (xB, yB), color, 2)
        cv2.putText(self.image_net, str(const.CLASSES[object.raw_label])+': '+str(round(object.confidence,1)), (xA,yA-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color, 2)
        # for each pedestrian show distance and velocity 
        # print("Inside draw_bbox function::: D: " +str(round(object.position[0],2))+","+str(round(object.position[1],2)))
        # cv2.putText(self.image_net, "D: (" +str(round(object.position[0],2))+","+str(round(object.position[1],2))+")", center_point, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,255,0), 2)
        # print("Inside draw_bbox function::: angle: " +str(round(angle,2)))
        cv2.putText(self.image_net, "angle: " +str(round(angle,2)), (center_point[0], center_point[1]+const.MAX_DEPTH), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,255,0), 2)
        if depth is not None:
            cv2.putText(self.image_net, "depth: " +str(round(depth,2)) + " m", center_point, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,255,0), 2)

    def collision_warning(self, objects, warning_list, display_resolution, camera_res, depth_map):
        # Access instance variable
        self.image_net = self.image_net

        obj_array = objects.object_list
        warning = const.NO_WARNING
        if len(obj_array) > 0:
            for obj in objects.object_list:
                if (obj.tracking_state != sl.OBJECT_TRACKING_STATE.OK) or (not np.isfinite(obj.position[0])) or (obj.id < 0):
                    continue

                color = (0,255,0)
                angle = np.arctan2(obj.velocity[0],obj.velocity[1])* 180 / np.pi          
    
                if( obj.raw_label in const.PERSONS_VEHICLES_CLASSES and obj.position[0] < const.MAX_DEPTH): ## for person and vehicles
                    if(obj.position[1] > const.LEFT_RIGHT_DISTANCE and angle > -170 and angle < -95):
                        color = (0,128,255)
                        warning = const.MID_WARNING
                    if(obj.position[1] < -const.LEFT_RIGHT_DISTANCE and angle > -85 and angle < -10 ):
                        color = (0,128,255)
                        warning = const.MID_WARNING
                    if(abs(obj.position[1]) <= const.LEFT_RIGHT_DISTANCE and abs(obj.position[0]) <= const.CAUTION_DISTANCE):
                        color = (0,128,255)
                        warning = const.MID_WARNING
                    if(abs(obj.position[1]) <= const.LEFT_RIGHT_DISTANCE and abs(obj.position[0]) < const.STOP_DISTANCE):
                        color = (0,0,255)
                        warning = const.URGENT_WARNING
                    self.draw_bbox(obj, color, depth_map)
                warning_list.append(warning)     
        else:
            warning_list.append(const.NO_WARNING)

        # Publish warning
        self.collision_warning_publish.publish(np.max(warning_list))
