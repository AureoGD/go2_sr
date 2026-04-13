from env.go2_env import Go2Env
from control.self_righting.time_based_solution.time_based_scheduler import SchedulerTB
from env.tasks.self_righting_task import SelfRightingTask
from env.env_factory import create_env
from es_framework.core.env_config import EnvConfig
from env.tasks.self_righting_scenario import SelfRightingScenario
import time

env_config = EnvConfig(env_class=Go2Env,
                       controller_class=SchedulerTB,
                       task_class=SelfRightingTask,
                       scene_path="sim/assets/unitree_go2/scene.xml",
                       urdf_path="sim/assets/unitree_go2/go2.urdf",
                       tpe_model_path="tpe/models/tpe_cnn/best_model.pt",
                       normalizer_params={
                           "joint_limits": 1,
                           "torque_limits": 1
                       },
                       render=True)


def main(_env_config):

    env, env_spec = create_env(_env_config)
    scene = SelfRightingScenario()

    action = 2
    tick = 0
    total_reward = 0
    scenario = scene.sample(ch=0)
    q0, r0, b0 = scenario["q0"], scenario["r0"], scenario["b0"]
    env.reset(q0=q0, r0=r0, b0=b0)
    end_sim = False
    time_now = time.time()
    while not end_sim:
        action = dummy_rule(tick)
        # action = 6
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        end_sim = terminated or truncated
        tick += 1
        print(time.time() - time_now)
    env.close()
    print(f"Total reward: {total_reward} - Tick: {tick}")


def dummy_rule(tick):
    if tick < 25:
        action = 0
    elif tick < 25 + 80:
        action = 1
    elif tick < 25 + 80 + 310:
        action = 7
    elif tick < 25 + 80 + 310 + 370:
        action = 8
    elif tick < 25 + 80 + 310 + 370 + 300:
        action = 9
    elif tick < 25 + 80 + 310 + 370 + 300 + 210:
        action = 10
    elif tick < 25 + 80 + 310 + 370 + 300 + 210 + 210:
        action = 6
    else:
        action = 0

    return action


def debug_rgc(tick):

    if tick < 20:
        action = 0
    elif tick < 150:
        action = 1  # go_safe
    elif tick < 150 + 250:
        action = 2  # prepare_cw
    elif tick < 150 + 250 + 150:
        action = 3  # roll_cw
    elif tick < 150 + 250 + 150 + 250:
        action = 4  # landing_cw
    elif tick < 150 + 250 + 150 + 250 + 270:
        action = 5  # prone
    elif tick < 150 + 250 + 150 + 250 + 270 + 300:
        action = 6  # standing_up
    else:
        action = 0


if __name__ == "__main__":
    main(env_config)
