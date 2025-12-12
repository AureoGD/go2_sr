import numpy as np
from environment.env_go2 import Go2Env

DT = 0.001


def run_test():
    # Create the environment
    env = Go2Env(rendering=True, max_step=100)

    for i in range(10):
        obs, info = env.reset()
        done = False
        total_reward = 0
        ep = 0

        while not done:
            action = 0
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep += 1
            total_reward += reward

        print("Episode finished.")
        print("Total reward:", total_reward)

    env.close()


if __name__ == "__main__":
    run_test()
