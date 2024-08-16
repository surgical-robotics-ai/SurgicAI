This folder includes the code for image-guided imitation learning. Use the command line below to launch a training:

```python
python3 <your_algo_name>_train.py --task_name "$TASK_NAME" --view_name "$VIEW_NAME" --trans_error "$TRANS_ERROR" --angle_error "$ANGLE_ERROR"
```

Command to launch evaluation script:

```python
python3 Task_evaluation_<your_algo_name>.py --task_name "$TASK_NAME" --view_name "$VIEW_NAME" --trans_error "$TRANS_ERROR" --angle_error "$ANGLE_ERROR"
```