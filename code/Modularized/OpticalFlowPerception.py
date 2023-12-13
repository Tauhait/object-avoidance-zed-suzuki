
import cv2
import numpy as np
import time
import rospy
import constant as const
from std_msgs.msg import Int32

class OpticalFlowPerception:
    """
    This class performs optical flow analysis on a live video stream and makes decisions based on the flow.
    """
    def __init__(self, camera_id=2):
        # Initialize ROS node
        rospy.init_node('perception_back_cam', anonymous=True)

        # Initialize publisher for optical flow decisions
        self.optical_flow_publish = rospy.Publisher('optical_flow', Int32, queue_size=1)

        # Open camera capture
        self.cap = cv2.VideoCapture(camera_id)

        # Check if camera opened successfully
        if not self.cap.isOpened():
            raise Exception("Cannot open camera")

        # Initialize variables
        self.prev_gray = None
        self.mask = None
        self.prev_frame_time = 0
        self.font = cv2.FONT_HERSHEY_SIMPLEX
    
    def matrix_mul(self, mat):
        """
        Helper function to calculate the sum of all elements in a matrix.
        """
        mat_shape = mat.shape
        total_elem = np.prod(mat_shape)
        reshaped_mat = np.reshape(mat, (total_elem, 1))
        result_vector = np.dot(reshaped_mat.T, np.ones((total_elem, 1)))
        result_vector = np.reshape(result_vector, (-1,))
        return result_vector
    
    def get_decision(self, flow):
        """
        Makes a decision based on the flow value.
        """
        if flow < (-1 * const.CLOSENESS_THRES):
            return const.TRAFFIC_FROM_LEFT
        elif flow > const.CLOSENESS_THRES:
            return const.TRAFFIC_FROM_RIGHT
        else:
            return const.SAFE_TO_OVERTAKE
    
    def get_decision_text(self, decision):
        """
        Returns text corresponding to the decision.
        """
        if decision == const.TRAFFIC_FROM_LEFT:
            return "LEFT"
        elif decision == const.TRAFFIC_FROM_RIGHT:
            return "RIGHT"
        else:
            return "OVERTAKE"
        
    def show_optical_flow(self, magnitude, angle):
        """
        Shows the optical flow as a colored image.
        """

        # Define HSV mask for color representation
        hsv = np.zeros_like(self.frame)
        hsv[..., 1] = 255

        # Convert flow direction to hue
        hsv[..., 0] = angle * 180 / np.pi / 2

        # Normalize and convert flow magnitude to value
        hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

        # Convert HSV to BGR for display
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Show optical flow image
        cv2.imshow("dense optical flow", bgr)

    def run(self):
        """
        Main loop for capturing frames, calculating optical flow, and making decisions.
        """
        while True:
            # Capture frame
            ret, frame = self.cap.read()

            # Check if frame captured successfully
            if not ret:
                print("Error: Failed to capture frame")
                break

            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Calculate optical flow
            if self.prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(self.prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

                # Convert flow to vector and normalize
                flow_ = int(self.matrix_mul(flow))
                magnitude_ = int(self.matrix_mul(magnitude))
                angle_ = int(self.matrix_mul(angle))

                # Make decision based on flow
                decision = self.get_decision(flow_)

                # Publish decision
                self.optical_flow_publish.publish(decision)

                # Show decision on image
                decision_text = self.get_decision_text(decision)
                cv2.putText(frame, decision_text, (7, 70), self.font, 1, (100, 255, 0), 1, cv2.LINE_AA)

                # Show optical flow image
                self.show_optical_flow(magnitude, angle)

            # Update previous frame
            self.prev_gray = gray

            # Show input frame
            cv2.imshow("input", frame)

            # Check for keyboard input
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release resources
        self.cap.release()
        cv2.destroyAllWindows()
