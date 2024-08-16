# Teleoperation Data-Collection Tutorial

This README provides instructions for collecting and processing dataset from the Master Tool Manipulator (MTM) in the Surgical Robotics Challenge (SRC) environment.

Launch ROS and SRC:

```bash
roscore
./ambf_simulator --launch_file <surgical_robotics_challenge>/launch.yaml -l 0,1,3,4,13,14 -p 200 -t 1 --override_max_comm_freq 120
```

Activate dVRK:
```bash
cd ./catkin_ws_dvrk/src/dvrk/dvrk_config_jhu/jhu-daVinci
rosrun dvrk_robot dvrk_console_json -j console-MTML-MTMR.json 
```

Connect MTM with SRC:
```bash
cd ~/surgical_robotics_challenge/scripts/surgical_robotics_challenge/teleoperation
./mtm_psm_pair_teleop.sh 
```
Turn on the data recorder:
```bash
python3 Demo_recorder_ambf.py
```

Replay the trajectory and convert it into training-compatible dataset:
```bash
python3 Trajectory_replayer.py
```
