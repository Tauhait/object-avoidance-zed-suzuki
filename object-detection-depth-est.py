import pyzed.sl as sl
import math
import numpy as np
import sys
import cv2
import torch
import matplotlib.path as mplPath
import socket
import time
from ultralytics import YOLO

FRAMES =  30
CONF_THRESHOLD = 100
TRANSLATION = (2.75, 4.0, 0)
DETECTION_CONF_THRESHOLD = 25
ROI_H = [0.05, 0.45, 0.54, 0.95]
ROI_W = [0.6, 0.21, 0.21, 0.6]
DEPTH = 2000
SAFE_DISTANCE = 25
MAX_SPEED = 10

classes = ['person', 'bicycle', 'car', 'motorcycle', 'none', 'none', 'none', 
           'none', 'none', 'traffic light', 'none', 'stop sign', 'none']

UDPsend_IP = "192.168.50.1"  # Mabx IP for sending from Pegasus
UDPsend_PORT = 30000  # Mabx port for sending from Pegasus

sockSend = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)



def getCheckSum(msg):
    cs = 0
    for m in msg:
        cs += m
    cs = (0x00 - cs) & 0x000000FF
    cs = cs & 0xFF
    return cs

def setSpeed(speed):
    speed = speed * 128
    H_speed = (int)(speed) >> 8
    L_speed = (int)(speed) & 0xff
    return H_speed, L_speed

def setAngle(m_nAngle, deltaAngle):
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

def sendToMabx(speed, m_nAngle, angle, flasher, counter):
    # speed = max(speed, 30)
    H_Angle, L_Angle = setAngle(m_nAngle, -1 * angle)
    H_Speed, L_Speed = setSpeed(speed)

    msg_list = [1, counter, 0, 1, 52, 136, 215, 1, H_Speed, L_Speed, H_Angle, L_Angle, 0, flasher, 0, 0, 0, 0]

    msg_list[2] = getCheckSum(msg_list)

    message = bytearray(msg_list)

    print("RC: ", message[1])
    print("Mode: ", message[7])
    print("Speed: ", message[8], message[9])
    print("Angle: ", message[10], message[11])
    print("Checksum: ", message[2])
    print("Flasher: ", message[13])
    print(" ".join(hex(m) for m in message))
    print("===============================================================================")
    addr = (UDPsend_IP, UDPsend_PORT)
    sockSend.sendto(message, addr)

def parseArg(arg_len, argv, param):
    if(arg_len > 1):
        if(".svo" in argv):
            # SVO input mode
            param.set_from_svo_file(sys.argv[1])
            print("Sample using SVO file input "+ sys.argv[1])
        elif(len(argv.split(":")) == 2 and len(argv.split(".")) == 4):
            #  Stream input mode - IP + port
            l = argv.split(".")
            ip_adress = l[0] + '.' + l[1] + '.' + l[2] + '.' + l[3].split(':')[0]
            port = int(l[3].split(':')[1])
            param.set_from_stream(ip_adress,port)
            print("Stream input mode")
        elif (len(argv.split(":")) != 2 and len(argv.split(".")) == 4):
            #  Stream input mode - IP
            param.set_from_stream(argv)
            print("Stream input mode")
        elif("HD2K" in argv):
            param.camera_resolution = sl.RESOLUTION.HD2K
            print("Using camera in HD2K mode")
        elif("HD1200" in argv):
            param.camera_resolution = sl.RESOLUTION.HD1200
            print("Using camera in HD1200 mode")
        elif("HD1080" in argv):
            param.camera_resolution = sl.RESOLUTION.HD1080
            print("Using camera in HD1080 mode")
        elif("HD720" in argv):
            param.camera_resolution = sl.RESOLUTION.HD720
            print("Using camera in HD720 mode")
        elif("SVGA" in argv):
            param.camera_resolution = sl.RESOLUTION.SVGA
            print("Using camera in SVGA mode")
        elif("VGA" in argv and "SVGA" not in argv):
            param.camera_resolution = sl.RESOLUTION.VGA
            print("Using camera in VGA mode")

def open_camera():
    # Create a Camera object
    zed = sl.Camera()
    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE  # Use PERFORMANCE depth mode
    init_params.coordinate_units = sl.UNIT.METER  # Use meter units (for depth measurements)
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    init_params.camera_fps = FRAMES
    
    if (len(sys.argv) > 1):
        parseArg(len(sys.argv), sys.argv[1], init_params)
    
    # Open the camera
    camera_open_err = zed.open(init_params)

    if camera_open_err != sl.ERROR_CODE.SUCCESS:
        print("Error! Camera cannot be opened or initialization error with object detection")
        exit(1)
    # Create and set RuntimeParameters after opening the camera
    runtime_parameters = sl.RuntimeParameters()
    #runtime_parameters.sensing_mode = sl.SENSING_MODE.STANDARD
    runtime_parameters.confidence_threshold = CONF_THRESHOLD
    runtime_parameters.texture_confidence_threshold = CONF_THRESHOLD
    return zed, runtime_parameters

def get_image_depth_pointcloud():
    image = sl.Mat()
    depth = sl.Mat()
    point_cloud = sl.Mat()
    return image, depth, point_cloud

def load_model(device):
    #Load YOLOv5 model
    #model = torch.hub.load('/usr/local/zed/samples/object-avoidance-zed-suzuki/pytorch_yolov8', model='yolov8',source='local', verbose=True, weights='yolov8.pt')   
    model = YOLO("yolov8m.pt")
    #model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, force_reload=True)
    model = model.to(device) 
    print("Model loaded")
    return model

def get_roi(img_w, img_h):
    roi = [ [ROI_H[0] * img_h, ROI_W[0] * img_w], 
            [ROI_H[1] * img_h, ROI_W[1] * img_w], 
            [ROI_H[2] * img_h, ROI_W[2] * img_w],
            [ROI_H[3] * img_h, ROI_W[3] * img_w] ]
    return roi

def get_detect_point_with_conf(box):
    object_class = int(box[5])
    if object_class < 4 or object_class == 11:
        xA = int(box[0])
        yA = int(box[1])
        xB = int(box[2])                
        yB = int(box[3])
        
        c1, c2 = (xA, yA), (xB, yB)
        bbox_height = yB - yA
        conf = round(box[4].item(), 2)
        #calculating center point of bounding box
        bbox_center_point = round((c1[0] + c2[0]) / 2), round((c1[1] + c2[1]) / 2)

        y = bbox_center_point[1] + (bbox_height / 2)
        detect_point = (bbox_center_point[0], y)
    return detect_point, conf, bbox_center_point, object_class

def get_distance_point_cloud(point_cloud_value):
    distance = math.sqrt(
        point_cloud_value[0] * point_cloud_value[0] +
        point_cloud_value[1] * point_cloud_value[1] +
        point_cloud_value[2] * point_cloud_value[2])             
    return round(distance, 2)

def draw_bbox(img, box):
    xA = int(box[0])
    yA = int(box[1])
    xB = int(box[2])                
    yB = int(box[3])
    # Draw bounding box    
    cv2.rectangle(img, (xA, yA), (xB, yB), (0, 255, 0), 2)

def draw_class_conf(img, box, cls, conf):
    xA = int(box[0])
    yA = int(box[1])
    # Class and confidence value at top of box
    cv2.putText(img, 
                str(classes[cls])+':'+str(conf), 
                (xA,yA), 
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,255,0), 1)

def draw_object_distance(img, depth, bbox_center_point):
    #Show distance of object at the center of object
    cv2.putText(img, 
                str(depth)+'m', 
                bbox_center_point, 
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,255,255), 1) 

def draw_roi(img, roi):
    cv2.polylines(img, np.array([roi], np.int32), True, (255, 0, 0), 2)

def main(device):
    zed, runtime_parameters = open_camera()
    image, depth, point_cloud = get_image_depth_pointcloud()
    model = load_model(device)
    while zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
        start_time = time.time()
        print(f"Time: {start_time}")
        zed.retrieve_image(image, sl.VIEW.RIGHT)        
        img_data_seq = image.get_data()
        (img_w, img_h) = (img_data_seq.shape[0], img_data_seq.shape[1])
        roi = get_roi(img_w, img_h)
        
        poly_path = mplPath.Path(roi)
        
        draw_roi(img_data_seq, roi)
        obj_detections = model(img_data_seq)
        
        zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU)
        
        depth_list = []
        depth_list.append(DEPTH)

        for box in obj_detections.xyxy[0]:
            detect_point, conf, bbox_center_point, object_class = get_detect_point_with_conf(box)
            if poly_path.contains_point(detect_point) and conf > DETECTION_CONF_THRESHOLD:
                _, point_cloud_value = point_cloud.get_value( bbox_center_point[0], 
                                                              bbox_center_point[1])
                depth = get_distance_point_cloud(point_cloud_value)
                if not np.isnan(depth) and not np.isinf(depth):
                   draw_object_distance(img_data_seq, depth, bbox_center_point)
                else:                    
                    depth = DEPTH
                    print("Can't estimate distance at this position.")
                    print("Your camera is probably too close to the scene, please move it backwards.\n")
                
                depth_list.append(depth)
                draw_bbox(img_data_seq, box)
                draw_class_conf(img_data_seq, box, object_class, conf) 

        if depth_list:  
            depth = min(depth_list)
        
        print('Depth: ', depth)
        speed = MAX_SPEED
        if (depth < SAFE_DISTANCE):
            speed = 0
            print("OBSTACLE DETECTED .........!!")
            print("BRAKING ........!!!")
        else:    
            speed = MAX_SPEED
            
        print("speed_value = ", speed)
        # init_angle = 0
        # delta_angle = 0
        # sendToMabx(s, init_angle, delta_angle, 0, counter)        
        # counter = (counter + 1) % 256

        img = cv2.resize(img, (700, 500))
        cv2.imshow("Object Avoidance", img)
        cv2.waitKey(2)
        time_taken = round((time.time() - start_time),2)
        print('(%s sec)'% (time_taken))

        #Press q to exit the detection
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
                
        
    cv2.destroyAllWindows()
    zed.close()
    print('camera closed')
    print('\nFINISHED')

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    with torch.no_grad():
        main(device)