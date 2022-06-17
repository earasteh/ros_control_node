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
                 max_steer=np.deg2rad(24), wheelbase=0.0):
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

        # self._waypoints = waypoints
        self._lookahead_distance = 5.0
        self.cross_track_deadband = 0.01

        # self.px = path_x
        # self.py = path_y
        # self.pyaw = path_yaw
    def find_target_path_id(self, px, py, x, y, yaw):
        L = self.L

        # Calculate position of the front axle
        fx = x + L * np.cos(yaw)
        fy = y + L * np.sin(yaw)

        dx = fx - px  # Find the x-axis of the front axle relative to the path
        dy = fy - py  # Find the y-axis of the front axle relative to the path

        d = np.hypot(dx, dy)  # Find the distance from the front axle to the path
        target_index = np.argmin(d)  # Find the shortest distance in the array

        return target_index, dx[target_index], dy[target_index], d[target_index], d

    def stanley_control(self, x, y, yaw, current_velocity, px, py, pyaw):
        """
        :param x:
        :param y:
        :param yaw:
        :param current_velocity:
        :return: steering output, target index, crosstrack error
        """
        target_index, dx, dy, absolute_error, _ = self.find_target_path_id(px, py, x, y, yaw)
        yaw_error = normalise_angle(pyaw[target_index] - yaw)
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

class ROS_Stanley:
    def __init__(self):
        rospy.init_node('Stanley_control')
        self.rate = rospy.Rate(50.0)
        
        self.px = [0]
        self.py = [0]
        self.pyaw = [0]
        
        self.x = 0
        self.y = 0
        self.yaw = 0
        
        self.vx = 0
        self.vy = 0
        self.vz = 0
        self.wx = 0
        self.wy = 0
        self.wz = 0
        
        self.controller = StanleyController(control_gain=2.5, softening_gain=1, yaw_rate_gain=0, steering_damp_gain=0,
                                        max_steer= np.deg2rad(10), wheelbase=2)
        
        rospy.Subscriber("/mpc_waypoints" , LaneArray, callback = self.LaneArrayCallback, queue_size=1)
        rospy.Subscriber("/current_pose"  , PoseStamped, callback = self.CurrentPoseCallback, queue_size=1)
        rospy.Subscriber("/vehicle_status", TwistStamped,callback = self.CurrentTwistCallback, queue_size=1)
        
        while not rospy.is_shutdown():
            self.publisher()
        
    def LaneArrayCallback(self, msg):
        """Get the x,y, yaw coordinates of reference path"""
        self.px = msg.lanes[0].waypoints[0].pose.pose.position.x
        self.py = msg.lanes[0].waypoints[0].pose.pose.position.y
        Qx = msg.lanes[0].waypoints[0].pose.pose.orientation.x
        Qy = msg.lanes[0].waypoints[0].pose.pose.orientation.y
        Qz = msg.lanes[0].waypoints[0].pose.pose.orientation.z
        Qw = msg.lanes[0].waypoints[0].pose.pose.orientation.w
        _, _, self.pyaw = euler_from_quaternion([Qx, Qy, Qz, Qw])
    
    def CurrentPoseCallback(self, msg):
        """Get the current x,y,yaw of the vehicle"""
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        xr = msg.pose.pose.orientation.x
        yr = msg.pose.pose.orientation.y
        zr = msg.pose.pose.orientation.z
        wr = msg.pose.pose.orientation.w
        _, _, self.yaw = euler_from_quaternion([xr, yr, zr, wr])
        
    def CurrentTwistCallback(self, msg):
        """Get the current vx,vy,vz,wx,wy,wz (linear and angular accelerations) of the vehicle"""
        self.vx = msg.twist.twist.linear.x
        self.vy = msg.twist.twist.linear.y
        self.vz = msg.twist.twist.linear.z
        self.wx = msg.twist.twist.angular.x
        self.wy = msg.twist.twist.angular.y
        self.wz = msg.twist.twist.angular.z
 
    def publisher(self):
        # msg_twist = TwistStamped()
        # msg_pose = PoseStamped()
        print('x, y, yaw =' + str(self.x) + ',' + str(self.y) + ',' + str(self.yaw))
        print('px, py, pyaw = '+ str(self.px)+ ',' + str(self.py) + ',' + str(self.pyaw))
        
        limited_steering_angle, target_index, crosstrack_error = self.controller.stanley_control(self.x, self.y,
                                                                                                 self.yaw, self.vx, 
                                                                                                 self.px, self.py, self.pyaw)
        
        msg_ctrl = ControlCommandStamped()
        msg_ctrl.cmd.steering_angle = limited_steering_angle
        
        print(limited_steering_angle)
        
        # pub_twist = rospy.Publisher("/twist_raw", TwistStamped, queue_size=1)
        pub_pose = rospy.Publisher("/ctrl_raw", ControlCommandStamped, queue_size=1)
        pub_pose.publish(msg_ctrl)


if __name__ == '__main__':
    try:
        ROS_Stanley()
    except rospy.ROSInterruptException:
        pass