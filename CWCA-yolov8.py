#!/usr/bin/env python3

import sys
import numpy as np
import time
import argparse
import torch
import cv2
import pyzed.sl as sl
from ultralytics import YOLO

from threading import Lock, Thread
from time import sleep

import ogl_viewer.viewer as gl
import cv_viewer.tracking_viewer as cv_viewer
import rospy
from std_msgs.msg import Float32, String

rospy.init_node('perception', anonymous=True)

Flag = rospy.Publisher('collision_warning', Float32, queue_size=1)

lock = Lock()
run_signal = False
exit_signal = False
MAX_CLASS_ID = 7

CLASSES = ['person', 'bicycle', 'car', 'motocycle', 'route board', 
           'bus', 'commercial vehicle', 'truck', 'traffic sign', 'traffic light',
            'autorickshaw','stop sign', 'ambulance', 'bench', 'construction vehicle',
            'animal', 'unmarked speed bump', 'marked speed bump', 'pothole', 'police vehicle',
            'tractor', 'pushcart', 'temporary traffic barrier', 'rumblestrips', 
            'traffic cone', 'pedestrian crossing']

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
        if class_id >= MAX_CLASS_ID:
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
 
def torch_thread(weights, img_size, conf_thres=0.2, iou_thres=0.45):
    global image_net, exit_signal, run_signal, detections

    print("Intializing Network...")

    model = YOLO(weights)

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

def draw_bbox(img, object, color):
    #for object in objects.object_list:
    xA = int(object.bounding_box_2d[0][0])
    yA = int(object.bounding_box_2d[0][1])
    xB = int(object.bounding_box_2d[2][0])
    yB = int(object.bounding_box_2d[2][1])

    c1, c2 = (xA, yA), (xB, yB) 
    center_point = round((c1[0] + c2[0]) / 2), round((c1[1] + c2[1]) / 2) ## center of object
    angle = np.arctan2(object.velocity[0],object.velocity[1])* 180 / np.pi
    #dist = math.sqrt(object.position[0]*object.position[0] + object.position[1]*object.position[1])
    #vel = math.sqrt(object.velocity[0]*object.velocity[0] + object.velocity[1]*object.velocity[1])

    cv2.rectangle(img, (xA, yA), (xB, yB), color, 2)
    cv2.putText(img, str(CLASSES[object.raw_label])+': '+str(round(object.confidence,2)), (xA,yA-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, color, 2)
    # for each pedestrian show distance and velocity 
    cv2.putText(img, "D: (" +str(round(object.position[0],2))+","+str(round(object.position[1],2))+")", center_point, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,255,0), 2)
    cv2.putText(img, "angle: " +str(round(angle,2)), (center_point[0], center_point[1]+20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,255,0), 2)
    return img

def collision_warning(objects, flag_list, display_resolution, camera_res):
    global image_net
    obj_array = objects.object_list
    print(str(len(obj_array))+" Object(s) detected\n")
    flag = 0
    ### when objects present then--->
    if len(obj_array) > 0:
        obj_flag = 1
        ## for each object detected in frame
        for obj in objects.object_list:             
            #print(classes[obj.raw_label])            
            if (obj.tracking_state != sl.OBJECT_TRACKING_STATE.OK) or (not np.isfinite(obj.position[0])) or (
                    obj.id < 0):
                continue

            color = (0,255,0)
            angle = np.arctan2(obj.velocity[0],obj.velocity[1])* 180 / np.pi          
            
            
            if( obj.raw_label<=7 and obj.position[0]<20): ## for person and vehicles
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
                    #print("inside")
                    color = (0,0,255)
                    flag = 2
                if(abs(obj.position[1]) <= 2 and abs(obj.position[0])>=15):
                    color = (0,128,255)
                    flag = 1
                image_net = draw_bbox(image_net, obj, color)
            flag_list.append(flag)
            #file.write(str(time.time())+","+str(obj.raw_label)+","+str(obj.id)+","+str(flag)+","+str(obj.position[0])+","+str(obj.position[1])+","+str(angle)+","+str(current_vel)+"\n")        
    else:        
        print("No object detected")
        flag_list.append(0)
        
    flag_frame = np.max(flag_list)
    print(flag_list)  
    print("collision_warning: ",flag_frame)
    Flag.publish(flag_frame)
    image_net = cv2.resize(image_net, (700, 500))

def main():
    global image_net, exit_signal, run_signal, detections

    capture_thread = Thread(target=torch_thread, kwargs={'weights': opt.weights, 'img_size': opt.img_size, "conf_thres": opt.conf_thres})
    capture_thread.start()

    print("Initializing Camera...")

    zed = sl.Camera()

    input_type = sl.InputType()
    if opt.svo is not None:
        input_type.set_from_svo_file(opt.svo)

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters(input_t=input_type, svo_real_time_mode=True)
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA  # QUALITY
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    init_params.depth_maximum_distance = 50

    runtime_params = sl.RuntimeParameters()
    status = zed.open(init_params)

    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit()

    image_left_tmp = sl.Mat()

    print("Initialized Camera")

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

            collision_warning(objects, [0], display_resolution, camera_res)

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
            #image_net_left_ocv = np.full((display_resolution.height, display_resolution.width, 4), [245, 239, 239, 255], np.uint8)
            #np.copyto(image_net_left_ocv, image_net)
            cv_viewer.render_2D(image_left_ocv, image_scale, objects, obj_param.enable_tracking)
            #global_image = cv2.hconcat([image_net, image_track_ocv])
            # Tracking view
            #track_view_generator.generate_view(objects, cam_w_pose, image_track_ocv, objects.is_tracked)

            cv2.imshow("ZED | 2D View and Birds View", image_net)
            key = cv2.waitKey(10)
            if key == 27:
                exit_signal = True
        else:
            exit_signal = True

    viewer.exit()
    exit_signal = True
    zed.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov8m.pt', help='model.pt path(s)')
    parser.add_argument('--svo', type=str, default=None, help='optional svo file')
    parser.add_argument('--img_size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf_thres', type=float, default=0.4, help='object confidence threshold')
    opt = parser.parse_args()

    with torch.no_grad():
        main()
