import pyzed.sl as sl
import math
import numpy as np
import sys
import cv2 as cv

FRAMES =  30
CONF_THRESHOLD = 100
TRANSLATION = (2.75, 4.0, 0)
DETECTION_CONF_THRESHOLD = 25

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


def main():
    # Create a Camera object
    zed = sl.Camera()
    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE  # Use PERFORMANCE depth mode
    init_params.coordinate_units = sl.UNIT.METER  # Use meter units (for depth measurements)
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    # Set initialization parameters
    detection_parameters = sl.ObjectDetectionParameters()
    detection_parameters.enable_tracking = True # Objects will keep the same ID between frames
    detection_parameters.enable_segmentation = True # Outputs 2D masks over detected objects
    # choose a detection model
    detection_parameters.detection_model = sl.OBJECT_DETECTION_MODEL.MULTI_CLASS_BOX_ACCURATE

    if (len(sys.argv) > 1):
        parseArg(len(sys.argv), sys.argv[1], init_params)
    
    # Open the camera
    open_err = zed.open(init_params)
    # Enable object detection with initialization parameters
    zed_error = zed.enable_object_detection(detection_parameters)

    if open_err != sl.ERROR_CODE.SUCCESS and zed_error != sl.ERROR_CODE.SUCCESS:
        print("Error! Camera cannot be opened or initialization error with object detection")
        exit(1)
    # Create and set RuntimeParameters after opening the camera
    runtime_parameters = sl.RuntimeParameters()
    runtime_parameters.confidence_threshold = CONF_THRESHOLD
    runtime_parameters.texture_confidence_threshold = CONF_THRESHOLD
    # Set runtime parameters
    detection_parameters_rt = sl.ObjectDetectionRuntimeParameters()
    detection_parameters_rt.detection_confidence_threshold = DETECTION_CONF_THRESHOLD

    image = sl.Mat()
    depth = sl.Mat()
    point_cloud = sl.Mat()

    mirror_ref = sl.Transform()
    mirror_ref.set_translation(sl.Translation(TRANSLATION[0], TRANSLATION[1], TRANSLATION[2]))
    # tr_np = mirror_ref.m
    while zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
        # Retrieve left image
        zed.retrieve_image(image, sl.VIEW.LEFT)
        # Retrieve depth map. Depth is aligned on the left image
        zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
        # Retrieve colored point cloud. Point cloud is aligned on the left image.
        zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)

        # Get and print distance value in mm at the center of the image
        # We measure the distance camera - object using Euclidean distance
        x = round(image.get_width() / 2)
        y = round(image.get_height() / 2)
        err, point_cloud_value = point_cloud.get_value(x, y)

        distance = math.sqrt(point_cloud_value[0] * point_cloud_value[0] +
                            point_cloud_value[1] * point_cloud_value[1] +
                            point_cloud_value[2] * point_cloud_value[2])

        # point_cloud_np = point_cloud.get_data() # invalid value encountered in cast
        # point_cloud_np.dot(tr_np)
        
        if np.isnan(distance) or np.isinf(distance):
            # Increment the loop
            print("Can't estimate distance at this position.")
            print("Your camera is probably too close to the scene, please move it backwards.\n")
            continue
        print("Distance to Camera at ({}, {}) (image center): {:1.3} m".format(x, y, distance), end="\r")
        objects = sl.Objects() # Structure containing all the detected objects
        zed.retrieve_objects(objects, detection_parameters_rt) # Retrieve the detected objects
        # Count the number of objects detected
        print("{} Object(s) detected".format(len(objects.object_list)))
        for object in objects.object_list:
            if object.confidence * 100 <  DETECTION_CONF_THRESHOLD:
                continue
            
            object_id = object.id # Get the object id
            object_position = object.position # Get the object position
            object_velocity = object.velocity # Get the object velocity
            object_tracking_state = object.tracking_state # Get the tracking state of the object
            object_2Dbbox = object.bounding_box_2d; # Get the 2D bounding box of the object
            object_3Dbbox = object.bounding_box; # Get the 3D Bounding Box of the object
            print("\nID: {}\nVelocity: {}".format(object_id, object_velocity))
            pos_x, pos_y, pos_z = object_position
            print("\n3D position : [{0},{1},{2}]".format(pos_x, pos_y, pos_z))
            print("\nBounding box 2D :")

            for it in object_2Dbbox:            
                print(" " + str(it), end='')
            
            print("\nBounding box 3D :")
            
            for it in object_3Dbbox:            
                print(" " + str(it), end='')            
            
            if object_tracking_state == sl.OBJECT_TRACKING_STATE.OK:
                print("\nObject {0} is tracked\n".format(object_id))
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
        print("<======================================================================>")
    
        sys.stdout.flush()        
    # Close the camera
    zed.close()

if __name__ == "__main__":
    main()