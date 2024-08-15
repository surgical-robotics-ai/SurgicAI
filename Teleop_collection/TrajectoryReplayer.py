import time
import os
import sys
from glob import glob
import rosbag
import gc
# from surgical_robotics_challenge.utils.task3_init import NeedleInitialization
from surgical_robotics_challenge.psm_arm import PSM
from surgical_robotics_challenge.ecm_arm import ECM
import PyKDL
from surgical_robotics_challenge.simulation_manager import SimulationManager
dynamic_path = os.path.abspath(__file__ + "/../../")
sys.path.append(dynamic_path)

def gripper_msg_to_jaw(msg):
    min = -0.1
    max = 0.51
    jaw_angle = msg.position[0] + min / (max - min)
    return jaw_angle


if __name__ == '__main__':
    data_folder = os.path.join(dynamic_path, "test_replay")  ## add rosbags here!
    save_folder = os.path.join(dynamic_path, "test_record")  ## folder to save images
    file_list = glob(os.path.join(data_folder, "*.bag"))
    rosbag_name = file_list[1]
    
    bag = rosbag.Bag(rosbag_name)
    topics = list(bag.get_type_and_topic_info()[1].keys())
    types = [val[0] for val in bag.get_type_and_topic_info()[1].values()]

    count = 0
    topics_name = []
    psm1_pos = []
    psm2_pos = []
    t_psm1 = []
    t_psm2 = []
    psm1_jaw = []
    psm2_jaw = []
    ecm_pos = []
    needle_pos = []

    ### new bag replay
    for topic, msg, t in bag.read_messages(topics=topics[11]):
        assert topic == "/ambf/env/Needle/State", "load incorrect topics for needle state"
        pose_msg = msg.pose
        needle_pos_temp = PyKDL.Frame(
        PyKDL.Rotation.Quaternion(
            pose_msg.orientation.x,
            pose_msg.orientation.y,
            pose_msg.orientation.z,
            pose_msg.orientation.w
        ),
        PyKDL.Vector(
            pose_msg.position.x/10.,
            pose_msg.position.y/10.,
            pose_msg.position.z/10.
        )
        )
        # break
        needle_pos.append(needle_pos_temp)
    #############################

    ### new bag replay
    for topic, msg, t in bag.read_messages(topics=topics[14]):
        assert topic == "/ambf/env/psm1/baselink/State", "load incorrect topics for psm 1 jp"
        # psm1_pos_temp = msg.joint_positions[0:6]
        psm1_pos_temp = [msg.joint_positions[0],
                         msg.joint_positions[1],
                         msg.joint_positions[2] / 10.,
                         msg.joint_positions[3],
                         msg.joint_positions[4],
                         msg.joint_positions[5]]
        psm1_pos.append(psm1_pos_temp)
        count += 1
    print("psm 1 record count: ", count)
    count = 0

    for topic, msg, t in bag.read_messages(topics=topics[0]):
        assert topic == "/MTML/gripper/measured_js", "load incorrect topics for psm 1 jaw"
        psm1_jaw_ambf_temp = gripper_msg_to_jaw(msg)
        psm1_jaw.append(psm1_jaw_ambf_temp)
        count += 1
    print("psm 1 jaw record count: ", count)
    count = 0

    # for topic, msg, t in bag.read_messages(topics=topics[15]):
    for topic, msg, t in bag.read_messages(topics=topics[16]):
        assert topic == "/ambf/env/psm2/baselink/State", "load incorrect topics for psm 2 jp"
        # psm1_pos_temp = msg.joint_positions[0:6]
        psm2_pos_temp = [msg.joint_positions[0],
                    msg.joint_positions[1],
                    msg.joint_positions[2] / 10.,
                    msg.joint_positions[3],
                    msg.joint_positions[4],
                    msg.joint_positions[5]]
        psm2_pos.append(psm2_pos_temp)
        count += 1
    print("psm 2 record count: ", count)
    count = 0

    for topic, msg, t in bag.read_messages(topics=topics[1]):
        assert topic == "/MTMR/gripper/measured_js", "load incorrect topics for psm 2 jaw"
        psm2_jaw_ambf_temp = gripper_msg_to_jaw(msg)
        psm2_jaw.append(psm2_jaw_ambf_temp)
        count += 1
    print("psm 2 jaw record count: ", count)
    count = 0

    gc.collect()

    simulation_manager = SimulationManager('record_test')
    time.sleep(0.5)
    w = simulation_manager.get_world_handle()
    time.sleep(0.2)
    w.reset_bodies()
    time.sleep(0.2)
    cam = ECM(simulation_manager, 'CameraFrame')
    # cam.servo_jp([0.0, 0.05, -0.01, 0.0])
    time.sleep(0.5)
    psm1 = PSM(simulation_manager, 'psm1', add_joint_errors=False)
    time.sleep(0.5)

    psm2 = PSM(simulation_manager, 'psm2', add_joint_errors=False)
    time.sleep(0.5)

    total_num = min(len(psm1_pos), len(psm2_pos), len(psm1_jaw), len(psm2_jaw))
    for i in range(total_num):
        cam.servo_jp(ecm_pos[i])
        psm1.servo_jp(psm1_pos[i])
        # psm1.set_jaw_angle(psm1_jaw[i] - 0.1)
        psm1.set_jaw_angle(psm1_jaw[i])
        psm2.servo_jp(psm2_pos[i])
        psm2.set_jaw_angle(psm2_jaw[i])
        time.sleep(0.01)
        count += 1
        sys.stdout.write(f"\r Run Progress: {count} / {total_num}")
        sys.stdout.flush()
