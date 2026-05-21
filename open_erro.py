import pickle
import os

with open("/home/CCT/9791086/mujoco_sr/failed_tasks/failed_task_3.pkl", "rb") as f:
    data = pickle.load(f)

print(data[4])
