#!/usr/bin/env python3
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

rospy.init_node('perception_front_cam', anonymous=True)

global lat, lon, lane_state 

collision_warning_publish = rospy.Publisher('collision_warning', Int32, queue_size=1)
collision_avoidance_publish = rospy.Publisher('collision_avoidance', Int32, queue_size=1)
overtake_lat_lon_publish = rospy.Publisher('overtake_lat_lon', String, queue_size=1)
lane_state_publish = rospy.Publisher('lane_state', Int32, queue_size=1)

lock = Lock()
red_pixels = []
green_pixels = []
show_plot = True
run_signal = False
exit_signal = False
train_id_to_color = [c.color for c in const.DRIVABLES if (c.train_id != -1 and c.train_id != 255)]
train_id_to_color = np.array(train_id_to_color)

def callback_lane_state(data):
    global lane_state 
    lane_state = data.data
rospy.Subscriber("/lane_state", Int8, callback_lane_state)

def callback_latlong(data):
    global lat,lon   
    lat = data.lat
    lon = data.lon
    
rospy.Subscriber("/novatel/oem7/bestpos",BESTPOS, callback_latlong)

def detections_to_custom_box(detections, im0):
    output = []
    for i, det in enumerate(detections):
        class_id = int(det.cls)
        if class_id not in const.REQ_CLASSES:
            continue
        
        xywh = det.xywh[0]

        # Creating ingestable objects for the ZED SDK
        obj = sl.CustomBoxObjectData()
        obj.bounding_box_2d = util.xywh2abcd(xywh, im0.shape)
        obj.label = class_id
        obj.probability = det.conf
        obj.is_grounded = False
        output.append(obj)
    return output
 
def torch_thread(weights, img_size, device, conf_thres=0.2, iou_thres=0.45):
    global image_net, exit_signal, run_signal, detections

    print("Inside torch_thread function:::Intializing Network...")

    model = YOLO(weights)
    model.to(device)
    print("Inside torch_thread function:::YOLO model loaded...")
    while not exit_signal:
        if run_signal:
            lock.acquire()

            img = cv2.cvtColor(image_net, cv2.COLOR_BGRA2RGB)
            det = model.predict(img, save=False, imgsz=img_size, conf=conf_thres, iou=iou_thres)[0].cpu().numpy().boxes

            # ZED CustomBox format (with inverse letterboxing tf applied)
            detections = detections_to_custom_box(det, image_net)
            lock.release()
            run_signal = False
        sleep(0.01)

def draw_bbox(object, color, depth_map):
    global image_net
    #for object in objects.object_list:
    xA = int(object.bounding_box_2d[0][0])
    yA = int(object.bounding_box_2d[0][1])
    xB = int(object.bounding_box_2d[2][0])
    yB = int(object.bounding_box_2d[2][1])

    c1, c2 = (xA, yA), (xB, yB) 
    h_x = 650
    h_y = 720
    center_point = round((c1[0] + c2[0]) / 2), round((c1[1] + c2[1]) / 2) ## center of object
    angle = util.get_angle_between_horizontal_base_object_center(h_x, center_point[0], h_x, image_net.shape[1], 
                                                            h_y, center_point[1], h_y, image_net.shape[0])
    #dist = math.sqrt(object.position[0]*object.position[0] + object.position[1]*object.position[1])
    #vel = math.sqrt(object.velocity[0]*object.velocity[0] + object.velocity[1]*object.velocity[1])

    depth = util.get_object_depth_val(center_point[0], center_point[1], depth_map)
    cv2.line(image_net, (h_x, h_y), (center_point[0], center_point[1]), color, 1)
    # cv2.line(image_net, (image_net.shape[1], image_net.shape[0]), (center_point[0], center_point[1]), color, 1)
    cv2.rectangle(image_net, (xA, yA), (xB, yB), color, 2)
    cv2.putText(image_net, str(const.CLASSES[object.raw_label])+': '+str(round(object.confidence,1)), (xA,yA-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color, 2)
    # for each pedestrian show distance and velocity 
    # print("Inside draw_bbox function::: D: " +str(round(object.position[0],2))+","+str(round(object.position[1],2)))
    # cv2.putText(image_net, "D: (" +str(round(object.position[0],2))+","+str(round(object.position[1],2))+")", center_point, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,255,0), 2)
    # print("Inside draw_bbox function::: angle: " +str(round(angle,2)))
    cv2.putText(image_net, "angle: " +str(round(angle,2)), (center_point[0], center_point[1]+const.MAX_DEPTH), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,255,0), 2)
    if depth is not None:
        cv2.putText(image_net, "depth: " +str(round(depth,2)) + " m", center_point, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,255,0), 2)
    
    #return img

def collision_warning(objects, warning_list, display_resolution, camera_res, depth_map):
    global image_net
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
                draw_bbox(obj, color, depth_map)
            warning_list.append(warning)     
    else:
        warning_list.append(const.NO_WARNING)

    collision_warning_publish.publish(np.max(warning_list))

def drivespace(depth_map):
    global red_pixels, green_pixels, show_plot, lane_state   
    freespace_frame = cv2.resize(cv2.cvtColor(image_net, cv2.COLOR_RGBA2RGB), (700, 500))

    pt_image = preprocess(freespace_frame)
    pt_image = pt_image.to(device)

    # get model prediction and convert to corresponding color
    y_pred = torch.argmax(PSPNet_model(pt_image.unsqueeze(0)), dim=1).squeeze(0)
    predicted_labels = y_pred.cpu().detach().numpy()

    cm_labels = (train_id_to_color[predicted_labels]).astype(np.uint8)
    
    green_masked_image, green_mask = util.get_green_masked_image(cm_labels)

    total_pixels = cm_labels.shape[0] * cm_labels.shape[1]
    num_green_pixels = np.sum(green_masked_image)
    normalized_num_green_pixels = num_green_pixels / total_pixels
    masked_image = np.zeros_like(cm_labels)
    driving_lane_space = int(normalized_num_green_pixels)
    
    if lane_state == const.DRIVING_LANE:
        red_masked_image, combined_red_mask = util.get_right_red_pixels(cm_labels)
        num_red_pixels = np.sum(red_masked_image)
        normalized_num_red_pixels = num_red_pixels / total_pixels
        combined_mask = green_mask | combined_red_mask
        overtake_lane_space = int(normalized_num_red_pixels)
        masked_image[combined_mask] = cm_labels[combined_mask]

        status, updated_mask, dist_to_free_lane_mid = util.is_clear_to_overtake(driving_lane_space, overtake_lane_space, green_masked_image, red_masked_image, masked_image, depth_map)
        if status == False or dist_to_free_lane_mid == None or np.isnan(dist_to_free_lane_mid):
            collision_avoidance_publish.publish(const.CONTINUE)
            overtake_lat_lon_publish.publish('0,0,0')
        else:
            print(f"dist_to_free_lane_mid: {dist_to_free_lane_mid}")
            masked_image = updated_mask
            overtake_bearing = util.get_bearing(dist_to_free_lane_mid)
            overtake_x, overtake_y = util.get_point_at_distance(lat, lon, dist_to_free_lane_mid, overtake_bearing, R=6371)
            collision_avoidance_publish.publish(const.OVERTAKE)
            overtake_lat_lon_publish.publish(f'{str(dist_to_free_lane_mid)},{str(overtake_x)},{str(overtake_y)}')
    elif lane_state == const.OVERTAKE_LANE:
        red_masked_image, combined_red_mask = util.get_left_red_pixels(cm_labels)
        num_red_pixels = np.sum(red_masked_image)
        normalized_num_red_pixels = num_red_pixels / total_pixels
        combined_mask = green_mask | combined_red_mask
        overtake_lane_space = int(normalized_num_red_pixels)
        masked_image[combined_mask] = cm_labels[combined_mask]
        if util.is_clear_to_switch(overtake_lane_space):
            collision_avoidance_publish.publish(const.CONTINUE)
            lane_state_publish.publish(const.DRIVING_LANE)

    
    return masked_image 

def main(device):
    global image_net, exit_signal, run_signal, detections, alternate_lane, lane_state
    alternate_lane = 0
    capture_thread = Thread(target=torch_thread, kwargs={'weights': opt.weights, 'img_size': opt.img_size, "conf_thres": opt.conf_thres, "device":device})
    capture_thread.start()

    print("Inside main function::: Initializing Camera...")
    zed = sl.Camera()
    print("Inside main function::: zed = sl.Camera() executed...")
    input_type = sl.InputType()
    print("Inside main function::: input_type.set_from_camera_id(1) executed...")
    if opt.svo is not None:
        input_type.set_from_svo_file(opt.svo)

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters(input_t=input_type, svo_real_time_mode=True)
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA  # QUALITY
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    init_params.depth_maximum_distance = 50

    runtime_params = sl.RuntimeParameters()
    print("Inside main function::: runtime_params = sl.RuntimeParameters() executed...")
    status = zed.open(init_params)
    print("Inside main function::: status = zed.open(init_params) executed...")

    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit()

    image_left_tmp = sl.Mat()
    print("Inside main function::: Initialized Camera")

    positional_tracking_parameters = sl.PositionalTrackingParameters()
    # If the camera is static, uncomment the following line to have better performances and boxes sticked to the ground.
    # positional_tracking_parameters.set_as_static = True
    zed.enable_positional_tracking(positional_tracking_parameters)

    obj_param = sl.ObjectDetectionParameters()
    obj_param.detection_model = sl.OBJECT_DETECTION_MODEL.CUSTOM_BOX_OBJECTS
    obj_param.enable_tracking = True
    zed.enable_object_detection(obj_param)

    objects = sl.Objects()
    obj_runtime_param = sl.ObjectDetectionRuntimeParameters()

    # Display
    camera_infos = zed.get_camera_information()
    camera_res = camera_infos.camera_configuration.resolution
    # Create OpenGL viewer
    viewer = gl.GLViewer()
    point_cloud_res = sl.Resolution(min(camera_res.width, 720), min(camera_res.height, 404))
    point_cloud_render = sl.Mat()
    viewer.init(camera_infos.camera_model, point_cloud_res, obj_param.enable_tracking)
    point_cloud = sl.Mat(point_cloud_res.width, point_cloud_res.height, sl.MAT_TYPE.F32_C4, sl.MEM.CPU)
    image_left = sl.Mat()
    # Utilities for 2D display
    display_resolution = sl.Resolution(min(camera_res.width, 1280), min(camera_res.height, 720))
    image_scale = [display_resolution.width / camera_res.width, display_resolution.height / camera_res.height]
    image_left_ocv = np.full((display_resolution.height, display_resolution.width, 4), [245, 239, 239, 255], np.uint8)

    # Utilities for tracks view
    camera_config = camera_infos.camera_configuration
    tracks_resolution = sl.Resolution(400, display_resolution.height)
    track_view_generator = cv_viewer.TrackingViewer(tracks_resolution, camera_config.fps, init_params.depth_maximum_distance)
    track_view_generator.set_camera_calibration(camera_config.calibration_parameters)
    image_track_ocv = np.zeros((tracks_resolution.height, tracks_resolution.width, 4), np.uint8)
    # Camera pose
    cam_w_pose = sl.Pose()
    depth_map = sl.Mat()
    
    lane_state = const.DRIVING_LANE
    while viewer.is_available() and not exit_signal:
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            # -- Get the image
            lock.acquire()
            zed.retrieve_image(image_left_tmp, sl.VIEW.LEFT)
            image_net = image_left_tmp.get_data()
            lock.release()
            run_signal = True

            # -- Detection running on the other thread
            while run_signal:
                sleep(0.001)

            # Wait for detections
            lock.acquire()
            # -- Ingest detections
            zed.ingest_custom_box_objects(detections)
            lock.release()
            zed.retrieve_objects(objects, obj_runtime_param)
            zed.retrieve_measure(depth_map, sl.MEASURE.DEPTH, sl.MEM.CPU)
            lane_state_publish.publish(lane_state)
            ########################################################################################
            overlay_image = drivespace(depth_map)
            collision_warning(objects, [0], display_resolution, camera_res, depth_map)
            #########################################################################################
            image_net = cv2.resize(image_net, (700, 500))          
            # -- Display
                  
            # Retrieve display data
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU, point_cloud_res)
            point_cloud.copy_to(point_cloud_render)
            zed.retrieve_image(image_left, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution)
            zed.get_position(cam_w_pose, sl.REFERENCE_FRAME.WORLD)

            # 3D rendering
            viewer.updateData(point_cloud_render, objects)
            # 2D rendering
            np.copyto(image_left_ocv, image_left.get_data())
            cv_viewer.render_2D(image_left_ocv, image_scale, objects, obj_param.enable_tracking)
            # Tracking view
            track_view_generator.generate_view(objects, cam_w_pose, image_track_ocv, objects.is_tracked)
            cv2.imshow("Collision Warning", image_net)
            cv2.imshow("Segmentation", overlay_image)
            key = cv2.waitKey(10)
            if key == 27:
                exit_signal = True
        else:
            exit_signal = True
    
    viewer.exit()
    exit_signal = True
    zed.close()


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Inside __main__::: " + str(device))
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov8m.pt', help='model.pt path(s)')
    parser.add_argument('--svo', type=str, default=None, help='optional svo file')
    parser.add_argument('--img_size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf_thres', type=float, default=0.7, help='object confidence threshold')
    opt = parser.parse_args()
    preprocess = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.485, 0.56, 0.406), std=(0.229, 0.224, 0.225))
                ])
    PSPNet_model = PSPNet(in_channels=3, num_classes=3, use_aux=True).to(device)
    pretrained_weights = torch.load(f'PSPNet_res50_20.pt', map_location="cpu")
    PSPNet_model.load_state_dict(pretrained_weights)
    PSPNet_model.eval()
    print("Inside __main__::: PSPNet_model loaded")
    lane_state_publish.publish(const.DRIVING_LANE)
    with torch.no_grad():
        main(device)

