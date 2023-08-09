#!/usr/bin/env python3

import sys
import numpy as np
import time
import argparse
import torch
import cv2
import pyzed.sl as sl
from ultralytics import YOLO
import math
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torchvision.transforms.functional as TF
from collections import namedtuple

# import segmentation model
from seg_model.pspnet import PSPNet

from threading import Lock, Thread
from time import sleep

import ogl_viewer.viewer as gl
import cv_viewer.tracking_viewer as cv_viewer
import rospy
from std_msgs.msg import Float32, String
import warnings
warnings.filterwarnings("ignore")

rospy.init_node('perception', anonymous=True)

Flag = rospy.Publisher('collision_warning', Float32, queue_size=1)

lock = Lock()
run_signal = False
exit_signal = False
MAX_CLASS_ID = 7
MAX_DEPTH = 20

CLASSES = ['person', 'bicycle', 'car', 'motocycle', 'route board', 
           'bus', 'commercial vehicle', 'truck', 'traffic sign', 'traffic light',
            'autorickshaw','stop sign', 'ambulance', 'bench', 'construction vehicle',
            'animal', 'unmarked speed bump', 'marked speed bump', 'pothole', 'police vehicle',
            'tractor', 'pushcart', 'temporary traffic barrier', 'rumblestrips', 'traffic cone', 'pedestrian crossing']
REQ_CLASSES = [0,1,2,3,4,6,7,8,9,10,11,12,13,15,16,17,18,19,20,26]
PERSONS_VEHICLES_CLASSES = [0,1,2,3,4,6,7,8,11,13,15,20,21]

Label = namedtuple( "Label", [ "name", "train_id", "color"])
drivables = [ 
             Label("direct", 0, (0, 255, 0)),        # green
             Label("alternative", 1, (0, 0, 255)),  # blue
             Label("background", 2, (0, 0, 0)),        # black          
            ]
train_id_to_color = [c.color for c in drivables if (c.train_id != -1 and c.train_id != 255)]
train_id_to_color = np.array(train_id_to_color)

def xywh2abcd(xywh, im_shape):
    output = np.zeros((4, 2))

    # Center / Width / Height -> BBox corners coordinates
    x_min = (xywh[0] - 0.5*xywh[2]) #* im_shape[1]
    x_max = (xywh[0] + 0.5*xywh[2]) #* im_shape[1]
    y_min = (xywh[1] - 0.5*xywh[3]) #* im_shape[0]
    y_max = (xywh[1] + 0.5*xywh[3]) #* im_shape[0]

    # A ------ B
    # | Object |
    # D ------ C

    output[0][0] = x_min
    output[0][1] = y_min

    output[1][0] = x_max
    output[1][1] = y_min

    output[2][0] = x_min
    output[2][1] = y_max

    output[3][0] = x_max
    output[3][1] = y_max
    return output

def get_class_label(det):
    # Assuming 'det' is a NumPy array, and you want to extract the first element
    number_str = str(det[0])

    # Now you can strip characters from the 'number_str'
    number_str = number_str.strip('[]').strip()

    # Convert the number string to an integer or use it as needed
    number = int(float(number_str))

    return number

def detections_to_custom_box(detections, im0):
    output = []
    for i, det in enumerate(detections):
        class_id = int(det.cls)
        if class_id not in REQ_CLASSES:
            continue
        
        xywh = det.xywh[0]

        # Creating ingestable objects for the ZED SDK
        obj = sl.CustomBoxObjectData()
        obj.bounding_box_2d = xywh2abcd(xywh, im0.shape)
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

def get_object_depth_val(x, y, depth_map):
    _, center_depth = depth_map.get_value(x, y, sl.MEM.CPU)
    if center_depth not in [np.nan, np.inf, -np.inf]:
        # print(f"Depth value at center: {center_depth} metres.")  
        return center_depth
    return None

def get_angle_between_horizontal_base_object_center(x1, x2, x3, x4, y1, y2, y3, y4):
    vec1 = (x2-x1, y2-y1)
    vec2 = (x4-x3, y4-y3)

    dot_prod = vec1[0] * vec2[0] + vec1[1] * vec2[1]
    magnitude_vec1 = math.sqrt(vec1[0]**2 + vec1[1]**2)
    magnitude_vec2 = math.sqrt(vec2[0]**2 + vec2[1]**2)

    cos_angle = dot_prod / (magnitude_vec1*magnitude_vec2)
    angle_rad = math.acos(cos_angle)
    angle_deg = math.degrees(angle_rad)
    if angle_deg > 180:
        angle_deg -= 360
    elif angle_deg < -180:
        angle_deg += 360
    return angle_deg

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
    angle = get_angle_between_horizontal_base_object_center(h_x, center_point[0], h_x, image_net.shape[1], 
                                                            h_y, center_point[1], h_y, image_net.shape[0])
    #dist = math.sqrt(object.position[0]*object.position[0] + object.position[1]*object.position[1])
    #vel = math.sqrt(object.velocity[0]*object.velocity[0] + object.velocity[1]*object.velocity[1])

    depth = get_object_depth_val(center_point[0], center_point[1], depth_map)
    cv2.line(image_net, (h_x, h_y), (center_point[0], center_point[1]), color, 1)
    # cv2.line(image_net, (image_net.shape[1], image_net.shape[0]), (center_point[0], center_point[1]), color, 1)
    cv2.rectangle(image_net, (xA, yA), (xB, yB), color, 2)
    cv2.putText(image_net, str(CLASSES[object.raw_label])+': '+str(round(object.confidence,1)), (xA,yA-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color, 2)
    # for each pedestrian show distance and velocity 
    # print("Inside draw_bbox function::: D: " +str(round(object.position[0],2))+","+str(round(object.position[1],2)))
    # cv2.putText(image_net, "D: (" +str(round(object.position[0],2))+","+str(round(object.position[1],2))+")", center_point, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,255,0), 2)
    # print("Inside draw_bbox function::: angle: " +str(round(angle,2)))
    cv2.putText(image_net, "angle: " +str(round(angle,2)), (center_point[0], center_point[1]+MAX_DEPTH), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,255,0), 2)
    if depth is not None:
        cv2.putText(image_net, "depth: " +str(round(depth,2)) + " m", center_point, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,255,0), 2)
    
    #return img

def collision_warning(objects, flag_list, display_resolution, camera_res, depth_map):
    global image_net
    obj_array = objects.object_list
    # print("Inside collision_warning function::: "+str(len(obj_array))+" Object(s) detected\n")
    flag = 0
    ### when objects present then--->
    if len(obj_array) > 0:
        obj_flag = 1
        ## for each object detected in frame
        for obj in objects.object_list:             
            # print(CLASSES[obj.raw_label])            
            if (obj.tracking_state != sl.OBJECT_TRACKING_STATE.OK) or (not np.isfinite(obj.position[0])) or (
                    obj.id < 0):
                continue

            color = (0,255,0)
            angle = np.arctan2(obj.velocity[0],obj.velocity[1])* 180 / np.pi          
            
            
            if( obj.raw_label in PERSONS_VEHICLES_CLASSES and obj.position[0] < MAX_DEPTH): ## for person and vehicles
                # ax.clear()
                # angle_list.append(angle)
                # ax.plot(angle_list)
                # plt.savefig("plot_angle.png")
                # print(obj.position[0], obj.position[1], angle)
                if(obj.position[1] > 2 and angle > -170 and angle < -95):
                    color = (0,128,255)
                    flag = 1
                if(obj.position[1] < -2 and angle > -85 and angle < -10 ):
                    color = (0,128,255)
                    flag = 1
                if(abs(obj.position[1]) <= 2 and abs(obj.position[0])<15):
                    color = (0,0,255)
                    flag = 2
                if(abs(obj.position[1]) <= 2 and abs(obj.position[0])>=15):
                    color = (0,128,255)
                    flag = 1
                draw_bbox(obj, color, depth_map)
            flag_list.append(flag)
            #file.write(str(time.time())+","+str(obj.raw_label)+","+str(obj.id)+","+str(flag)+","+str(obj.position[0])+","+str(obj.position[1])+","+str(angle)+","+str(current_vel)+"\n")        
    else:        
        # print("Inside collision_warning function::: No object detected")
        flag_list.append(0)
        
    flag_frame = np.max(flag_list)
    #print(flag_list)  
    # print("Inside collision_warning function::: collision_warning: ",flag_frame)
    Flag.publish(flag_frame)

def drivespace():
############################################################################################ PSP NET            
    freespace_frame = cv2.resize(cv2.cvtColor(image_net, cv2.COLOR_RGBA2RGB), (700, 500))

    pt_image = preprocess(freespace_frame)
    pt_image = pt_image.to(device)

    # get model prediction and convert to corresponding color
    y_pred = torch.argmax(PSPNet_model(pt_image.unsqueeze(0)), dim=1).squeeze(0)
    predicted_labels = y_pred.cpu().detach().numpy()

    cm_labels = (train_id_to_color[predicted_labels]).astype(np.uint8)
    green_mask = (cm_labels[:,:,1] > 0) & (cm_labels[:,:,0] == 0) & (cm_labels[:,:,2] == 0)
    
    green_masked_image = np.zeros_like(cm_labels)
    green_masked_image[green_mask] = cm_labels[green_mask]

    total_pixels = cm_labels.shape[0] * cm_labels.shape[1]
    num_green_pixels = np.sum(green_masked_image)
    normalized_num_green_pixels = num_green_pixels / total_pixels
    print(f"\n\nTotal green space available in pixels: {normalized_num_green_pixels}\n")
    height, width, _ = cm_labels.shape
    red_mask = (cm_labels[:,:,2] > 0) & (cm_labels[:,:,1] == 0) & (cm_labels[:,:,0] == 0)
    # Create a mask for the left side of the matrix
    left_mask = np.zeros((height, width // 2), dtype=bool)
    # Combine the left mask and the red mask for the right half
    combined_red_mask = np.hstack((left_mask, red_mask[:, width // 2:]))   
    red_masked_image = np.zeros_like(cm_labels)
    red_masked_image[combined_red_mask] = cm_labels[combined_red_mask]
    num_red_pixels = np.sum(red_masked_image)
    normalized_num_red_pixels = num_red_pixels / total_pixels
    print(f"\n\nTotal right red space available in pixels: {normalized_num_red_pixels}\n")
    combined_mask = green_mask | combined_red_mask
    masked_image = np.zeros_like(cm_labels)
    masked_image[combined_mask] = cm_labels[combined_mask]     
    return masked_image
#########################################################################################################   

def main(device):
    global image_net, exit_signal, run_signal, detections

    capture_thread = Thread(target=torch_thread, kwargs={'weights': opt.weights, 'img_size': opt.img_size, "conf_thres": opt.conf_thres, "device":device})
    capture_thread.start()

    print("Inside main function::: Initializing Camera...")

    zed = sl.Camera()

    input_type = sl.InputType()
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
    status = zed.open(init_params)

    if status != sl.ERROR_CODE.SUCCESS:
        # print(repr(status))
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
            ########################################################################################
            overlay_image = drivespace()
            collision_warning(objects, [0], display_resolution, camera_res, depth_map)
            #########################################################################################
            image_net = cv2.resize(image_net, (700, 500))
            # print("Inside main function::: image_net shape " + str(image_net.shape))
            ######################################################################################### get DEPTH center pixel           
            # -- Display
            
   
            #########################################################################################       
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
            cv2.imshow("Segmetation", overlay_image)
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
    with torch.no_grad():
        main(device)
