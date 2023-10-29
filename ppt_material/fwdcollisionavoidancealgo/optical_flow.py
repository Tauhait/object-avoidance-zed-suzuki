import cv2 as cv
import numpy as np
import time
import rospy
import const
from std_msgs.msg import Float32, String, Int8, Int32

cap = cv.VideoCapture(2)

rospy.init_node('perception_back_cam', anonymous=True)
optical_flow_publish = rospy.Publisher('optical_flow', Int32, queue_size=1)

def matrix_mul(mat):
    mat_shap = mat.shape
    total_elem = np.prod(mat_shap)
    reshaped_mat = np.reshape(mat, (total_elem, 1))
    result_vector = np.dot(reshaped_mat.T, np.ones((total_elem,1)))
    result_vector = np.reshape(result_vector, (-1,))
    return result_vector

def sigmoid(x):
    return 1/(1 + np.exp(-x))

if cap.isOpened():
    ret, frame = cap.read()  #capture a frame from live video
    prev_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    mask = np.zeros_like(frame)
    # Sets image saturation to maximum
    mask[..., 1] = 255
    #check whether frame is successfully captured
    # font which we will be using to display FPS
    prev_frame_time = 0
    font = cv.FONT_HERSHEY_SIMPLEX
    if ret:
        # continue to display window until 'q' is pressed
        minute_count = 0
        while True:
            minute_count += 1
            ret, frame = cap.read()
            # Opens a new window and displays the input frame
            cv.imshow("input", frame)            
            # Converts each frame to grayscale - we previously
            # only converted the first frame to grayscale
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            # time when we finish processing for this frame
            new_frame_time = time.time()            
            # fps will be number of frame processed in given time frame
            # Calculates dense optical flow by Farneback method
            flow = cv.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)            
            
            # Computes the magnitude and angle of the 2D vectors
            magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])            
            
            # Sets image hue according to the optical flow direction            
            mask[..., 0] = angle * 180 / np.pi / 2
            
            # Sets image value according to the optical flow magnitude (normalized)
            mask[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)
            
            # Converts HSV to RGB (BGR) color representation
            rgb = cv.cvtColor(mask, cv.COLOR_HSV2BGR)

            # Opens a new window and displays the output frame            
            flow_ = int(matrix_mul(flow))
            magnitude_ = int(matrix_mul(magnitude))
            angle_ = int(matrix_mul(angle))
            
            if(flow_ < (-1 * const.CLOSENESS_THRES)):
                decision = 1
            elif(flow_ > const.CLOSENESS_THRES):
                decision = -1
            else:
                decision = 0

            optical_flow_publish.publish(decision)
            decision_text = ''
            if decision == 1:
                decision_text = "LEFT_SIDE"
            elif decision == -1:
                decision_text = "RIGHT_SIDE"
            else:
                decision_text = "SAFE_TO_OVERTAKE"
            cv.putText(rgb, decision_text, (7, 70), font, 1, (100, 255, 0), 1, cv.LINE_AA)
            cv.imshow("dense optical flow", rgb)
            
            # Updates previous frame
            prev_gray = gray
            
            # Frames are read by intervals of 1 millisecond. The
            # programs breaks out of the while loop when the
            # user presses the 'q' key
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

    #print error if frame capturing was unsuccessful
    else:
        print("Error : Failed to capture frame")

# print error if the connection with camera is unsuccessful
else:
    print("Cannot open camera")

# The following frees up resources and
# closes all windows
cap.release()
cv.destroyAllWindows()