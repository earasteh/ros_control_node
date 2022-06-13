import numpy as np
from math import atan2, sin, cos
import rospy
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from autoware_msgs.msg import ControlCommandStamped, LaneArray, VehicleStatus
from geometry_msgs.msg import TwistStamped, PoseStamped


"""
:param angle:       (float) angle [rad]
:return angle:      (float) angle [rad]
"""
normalise_angle = lambda angle : atan2(sin(angle), cos(angle))

class StanleyController:
    def __init__(self, control_gain=2.5, softening_gain=1.0, yaw_rate_gain=0.0, steering_damp_gain=0.0,
                 max_steer=np.deg2rad(24), wheelbase=0.0,
                 waypoints=None):
        """
        Stanley Controller
        At initialisation
        :param control_gain:                (float) time constant [1/s]
        :param softening_gain:              (float) softening gain [m/s]
        :param yaw_rate_gain:               (float) yaw rate gain [rad]
        :param steering_damp_gain:          (float) steering damp gain
        :param max_steer:                   (float) vehicle's steering limits [rad]
        :param wheelbase:                   (float) vehicle's wheelbase [m]
        :param path_x:                      (numpy.ndarray) list of x-coordinates along the path
        :param path_y:                      (numpy.ndarray) list of y-coordinates along the path
        :param path_yaw:                    (numpy.ndarray) list of discrete yaw values along the path
        :param dt:                          (float) discrete time period [s]
        At every time step
        :param x:                           (float) vehicle's x-coordinate [m]
        :param y:                           (float) vehicle's y-coordinate [m]
        :param yaw:                         (float) vehicle's heading [rad]
        :param target_velocity:             (float) vehicle's velocity [m/s]
        :param steering_angle:              (float) vehicle's steering angle [rad]
        :return limited_steering_angle:     (float) steering angle after imposing steering limits [rad]
        :return target_index:               (int) closest path index
        :return crosstrack_error:           (float) distance from closest path index [m]
        """

        self.k = control_gain
        self.k_soft = softening_gain
        self.k_yaw_rate = yaw_rate_gain
        self.k_damp_steer = steering_damp_gain
        self.max_steer = max_steer
        self.L = wheelbase

        self._waypoints = waypoints
        self._lookahead_distance = 5.0
        self.cross_track_deadband = 0.01

        # self.px = path_x
        # self.py = path_y
        # self.pyaw = path_yaw
    def find_target_path_id(px, py, x, y, yaw, param):
        L = param.L

        # Calculate position of the front axle
        fx = x + L * np.cos(yaw)
        fy = y + L * np.sin(yaw)

        dx = fx - px  # Find the x-axis of the front axle relative to the path
        dy = fy - py  # Find the y-axis of the front axle relative to the path

        d = np.hypot(dx, dy)  # Find the distance from the front axle to the path
        target_index = np.argmin(d)  # Find the shortest distance in the array

        return target_index, dx[target_index], dy[target_index], d[target_index], d

    def stanley_control(self, x, y, yaw, current_velocity):
        """
        :param x:
        :param y:
        :param yaw:
        :param current_velocity:
        :return: steering output, target index, crosstrack error
        """
        target_index, dx, dy, absolute_error, _ = self.find_target_path_id(self.px, self.py, x, y, yaw, self.params)
        yaw_error = normalise_angle(self.pyaw[target_index] - yaw)
        # calculate cross-track error
        front_axle_vector = [np.sin(yaw), -np.cos(yaw)]
        nearest_path_vector = [dx, dy]
        crosstrack_error = np.sign(np.dot(nearest_path_vector, front_axle_vector)) * absolute_error
        crosstrack_steering_error = np.arctan2((self.k * crosstrack_error), (self.k_soft + current_velocity))

        desired_steering_angle = yaw_error + crosstrack_steering_error
        # Constrains steering angle to the vehicle limits
        limited_steering_angle = np.clip(desired_steering_angle, -self.max_steer, self.max_steer)

        return limited_steering_angle, target_index, crosstrack_error




# subscriber and publisher for the controller
# Subscribe: 
"""
/mpc_waypoints : reference waypoints (generated in mpc_waypoints_converter)
/current_pose : self pose
/vehicle_status : vehicle information (as velocity and steering angle source)
"""
# Publish:
"""
/twist_raw : command for vehicle
/ctrl_raw : command for vehicle
"""

def LaneArrayCallback(msg):
    px = msg.lanes[0].waypoints[0].pose.pose.position.x
    py = msg.lanes[0].waypoints[0].pose.pose.position.y
    Qx = msg.lanes[0].waypoints[0].pose.pose.orientation.x
    Qy = msg.lanes[0].waypoints[0].pose.pose.orientation.y
    Qz = msg.lanes[0].waypoints[0].pose.pose.orientation.z
    Qw = msg.lanes[0].waypoints[0].pose.pose.orientation.w
    _, _, pyaw = euler_from_quaternion([Qx, Qy, Qz, Qw])
    
    return px, py, pyaw
    
# def CurrCallback(self, msg):
#     # yaw = msg.

# def statCallback(self, msg):


def subscriber():
    rospy.Subscriber("/mpc_waypoints" , LaneArray, callback = LaneArrayCallback, queue_size=1)
    # rospy.Subscriber("/current_pose"  , PoseStamped, CurrCallback, queue_size=1)
    # rospy.Subscriber("/vehicle_status", TwistStamped,statCallback, queue_size=1)
    
    return px, py, pyaw
    
def publisher(px, py, pyaw):
    # msg_twist = TwistStamped()
    # msg_pose = PoseStamped()
    msg_ctrl = ControlCommandStamped()
    msg_ctrl.cmd.steering_angle = -1 * np.pi/180
    
    # pub_twist = rospy.Publisher("/twist_raw", TwistStamped, queue_size=1)
    pub_pose = rospy.Publisher("/ctrl_raw", ControlCommandStamped, queue_size=1)
    pub_pose.publish(msg_ctrl)


if __name__ == '__main__':
    global px, py, pyaw
    px = []
    py = []
    pyaw = []
    
    try:
        rospy.init_node('Stanley_control')
        while not rospy.is_shutdown():
            px,py,pyaw=subscriber()
            publisher(px, py, pyaw)
    except rospy.ROSInterruptException:
        pass