import numpy as np
from PyKDL import Frame, Rotation, Vector
from surgical_robotics_challenge.kinematics.psmIK import *
from surgical_robotics_challenge.kinematics.psmFK import *
import imitation.data.types as idt

def interpolate_actions(obs_array, max_action_diff):
    interpolated_obs = [obs_array[0]] 
    for i in range(1, len(obs_array)):
        current_obs = obs_array[i - 1]
        next_obs = obs_array[i]
        action_diff = next_obs - current_obs
        action_diff_psm2 = action_diff[7:13]
        max_steps_needed = np.ceil(np.max(np.abs(action_diff_psm2)/ max_action_diff)).astype(int)
        for step in range(1, max_steps_needed + 1):
            fraction = step / max_steps_needed
            intermediate_obs = current_obs + fraction * action_diff
            interpolated_obs.append(intermediate_obs)
    
    interpolated_obs = np.array(interpolated_obs)
    return interpolated_obs

def filter_actions(obs_array, max_action_diff, max_removals=100):
    removal_threshold = max_action_diff / 2
    filtered_obs = []
    i = 0

    while i < len(obs_array):
        current_obs = obs_array[i]
        filtered_obs.append(current_obs)
        removal_count = 0
        for j in range(i + 1, min(i + max_removals + 1, len(obs_array))):
            next_obs = obs_array[j]
            action_diff = np.abs(next_obs - current_obs)

            if np.all(action_diff[7:13] < removal_threshold):
                removal_count += 1
                continue
            else:
                break
        i += removal_count + 1

    return np.array(filtered_obs)

def data_processing(psm1_pos, psm2_pos, psm1_jaw, psm2_jaw, needle_pose, task_idx,T_w_b=None):
    for i in range(len(psm1_jaw)):
        if psm1_jaw[i] < 0:
            psm1_jaw[i] = 0
    for i in range(len(psm2_jaw)):
        if psm2_jaw[i] < 0:
            psm2_jaw[i] = 0
    total_num = min(len(psm1_pos), len(psm2_pos), len(psm1_jaw), len(psm2_jaw))
    obs_array = []
    act_array = []
    info_array = []
    Radius = 0.1018
    T_bmINn = Frame(Rotation.RPY(0., 0., -np.pi/6), Vector(-Radius*np.cos(np.pi/6), Radius*np.sin(np.pi/6), 0.)/10.0)

    for i in range(total_num):
        psm1_mat_temp = compute_FK(psm1_pos[i],7)
        psm1_frame_temp = convert_mat_to_frame(psm1_mat_temp)
        psm1_obs_temp = Frame2obs(psm1_frame_temp,psm1_jaw[i])

        psm2_mat_temp = compute_FK(psm2_pos[i],7)
        psm2_frame_temp = convert_mat_to_frame(psm2_mat_temp)
        psm2_obs_temp = Frame2obs(psm2_frame_temp,psm2_jaw[i])      

        assert (T_w_b is not None), "T_w_b is None"
        needle_obs = needle_goal_evaluator(needle_pose[i]*T_bmINn,T_w_b)
        needle_world = Frame2obs(needle_pose[i]*T_bmINn)


        obs_array_temp = np.concatenate((psm1_obs_temp,psm2_obs_temp,needle_obs,needle_world,task_idx))
        obs_array.append(obs_array_temp)
        
    max_action_diff = np.array([0.0005, 0.0005, 0.0005, np.deg2rad(2), np.deg2rad(2), np.deg2rad(2)])
    print("\n")
    print(f"original length: {len(obs_array)}")
    obs_array = interpolate_actions(obs_array, max_action_diff)
    print(f"interpolated length: {len(obs_array)}")
    obs_array = filter_actions(obs_array, max_action_diff)
    print(f"filtered length: {len(obs_array)}")

    obs_array = np.array(obs_array, dtype=np.float32) 
    act_array = np.diff(obs_array, axis=0).astype(np.float32)
    info_array = np.array([{"is_success":False} for _ in range(len(act_array))], dtype=object)
    info_array[-1] = {"is_success":True}
    terminal = True
    trajectory = idt.Trajectory(
        obs=obs_array,
        acts=act_array,
        infos=info_array,
        terminal=terminal
    )
    return trajectory

def Frame2obs(psm_pos,jaw=None):
    X_goal = psm_pos.p.x()
    Y_goal = psm_pos.p.y()
    Z_goal = psm_pos.p.z()
    rot_goal = psm_pos.M
    roll_goal,pitch_goal,yaw_goal  = rot_goal.GetRPY()
    if (roll_goal <= np.deg2rad(-360)):
        roll_goal += 2*np.pi
    elif (roll_goal >= np.deg2rad(0)):
        roll_goal -= 2*np.pi
    if jaw is not None:
        array_goal = np.array([X_goal,Y_goal,Z_goal,roll_goal,pitch_goal,yaw_goal,jaw],dtype=np.float32)
    else:
        array_goal = np.array([X_goal,Y_goal,Z_goal,roll_goal,pitch_goal,yaw_goal],dtype=np.float32)
    return array_goal

    
def convert_frame_to_mat(frame):
    np_mat = np.mat([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]], dtype=float)
    for i in range(3):
        for j in range(3):
            np_mat[i, j] = frame.M[(i, j)]

    for i in range(3):
        np_mat[i, 3] = frame.p[i]

    return np_mat


def convert_mat_to_frame(mat):
    frame = Frame(Rotation.RPY(0, 0, 0), Vector(0, 0, 0))
    for i in range(3):
        for j in range(3):
            frame[(i, j)] = mat[i, j]

    for i in range(3):
        frame.p[i] = mat[i, 3]

    return frame

def needle_goal_evaluator(grasp_in_World,T_w_b,lift_height = 0.010):

    lift_in_grasp_rot = Rotation(1, 0, 0,
                                0, 1, 0,
                                0, 0, 1)    
    lift_in_grasp_trans = Vector(0,0,lift_height)
    lift_in_grasp = Frame(lift_in_grasp_rot,lift_in_grasp_trans)

    gripper_in_lift_rot = Rotation(0, -1, 0,
                                -1, 0, 0,
                                0, 0, -1)

    gripper_in_lift_trans = Vector(0.0,0.0,0.0)
    gripper_in_lift = Frame(gripper_in_lift_rot,gripper_in_lift_trans)

    gripper_in_world = grasp_in_World*lift_in_grasp*gripper_in_lift
    gripper_in_base = T_w_b*gripper_in_world
    
    array_goal_base = Frame2obs(gripper_in_base)
    array_goal_base = np.append(array_goal_base,0.0)
    return array_goal_base