from PyKDL import Frame, Rotation, Vector
import numpy as np
import rospy
from ambf_msgs.msg import RigidBodyState

def pose_msg_to_frame(msg):
    """

    :param msg:
    :return:
    """
    p = Vector(msg.position.x,
               msg.position.y,
               msg.position.z)

    R = Rotation.Quaternion(msg.orientation.x,
                            msg.orientation.y,
                            msg.orientation.z,
                            msg.orientation.w)

    return Frame(R, p)

class NeedleKinematics_v2:
    # # Base in Needle Origin
    # T_bINn = Frame(Rotation.RPY(0., 0., 0.), Vector(-0.102, 0., 0.)/10.0)
    # # Mid in Needle Origin
    # T_mINn = Frame(Rotation.RPY(0., 0., -1.091), Vector(-0.048, 0.093, 0.)/10.0)
    # # Tip in Needle Origin
    # T_tINn = Frame(Rotation.RPY(0., 0., -0.585), Vector(0.056, 0.085, 0.)/10.0)

    # Base in Needle Origin (Modifoed Version)
    Radius = 0.1018
    T_bINn = Frame(Rotation.RPY(0., 0., 0.), Vector(-Radius, 0., 0.)/10.0)
    # Mid in Needle Origin
    T_mINn = Frame(Rotation.RPY(0., 0., -np.pi/3), Vector(-Radius*np.cos(np.pi/3), Radius*np.sin(np.pi/3), 0.)/10.0)
    # Tip in Needle Origin
    T_tINn = Frame(Rotation.RPY(0., 0., -np.pi/3*2), Vector(-Radius*np.cos(np.pi/3*2),Radius*np.sin(np.pi/3*2), 0.)/10.0)

    # base-mid center in Needle Origin
    T_bmINn = Frame(Rotation.RPY(0., 0., -np.pi/6), Vector(-Radius*np.cos(np.pi/6), Radius*np.sin(np.pi/6), 0.)/10.0)


    def __init__(self):
        """

        :return:
        """
        self._needle_sub = rospy.Subscriber(
            '/ambf/env/Needle/State', RigidBodyState, self.needle_cb, queue_size=1)
        # Needle in World
        self._T_nINw = Frame()

    def needle_cb(self, msg):
        """ needle callback; called every time new msg is received
        :param msg:
        :return:
        """
        self._T_nINw = pose_msg_to_frame(msg.pose)
        self._T_nINw.p = self._T_nINw.p /10.0

    def get_tip_pose(self):
        """

        :return:
        """
        T_tINw = self._T_nINw * self.T_tINn
        return T_tINw

    def get_base_pose(self):
        """

        :return:
        """
        T_bINw = self._T_nINw * self.T_bINn
        return T_bINw

    def get_mid_pose(self):
        """

        :return:
        """
        T_mINw = self._T_nINw * self.T_mINn
        return T_mINw

    def get_pose(self):
        return self._T_nINw

    def get_bm_pose(self):
        return self._T_nINw * self.T_bmINn
    
    def get_pose_angle(self,angle_degree):
        angle_rad = np.deg2rad(angle_degree)
        T_angle = Frame(Rotation.RPY(0., 0., -angle_rad), Vector(-self.Radius*np.cos(angle_rad), self.Radius*np.sin(angle_rad), 0.)/10.0)
        return self._T_nINw * T_angle

    def get_random_grasp_point(self,random_degree=None):
        Radius = 0.1018
        min_degree = 10
        max_degree = 50
        if (random_degree is None):
            random_degree = np.random.uniform(min_degree, max_degree)
            random_radian = np.deg2rad(random_degree)
        else:
            if not (min_degree <= random_degree <= max_degree):
                raise ValueError("random_degree out of range. Must be between 10 and 50 degrees.")
        T_randomINn = Frame(Rotation.RPY(0., 0., -random_radian), Vector(-Radius*np.cos(random_radian), Radius*np.sin(random_radian), 0.)/10.0)

        return self._T_nINw * T_randomINn