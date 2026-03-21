from environment.go2_sim import Go2ModelSimMuJoCo
from tpe.phases import Phase
import numpy as np
import random
import os

q0_base = np.array([0.7, 1.0, -2.1, 0, 1.4, -1.75, 0, 1.4, -1.8, 0, 1.4, -2.0])
render_sim = False
ep_num = 2500
directory_save = os.path.join("tpe", "data", "raw")


class TimeSchedulling():
    """
    Logical phase order:
        0 -> Hold
        1 -> GoSafe
        2 -> Prepare
        3 -> Roll
        4 -> Landing
        5 -> Prone
        6 -> StandUp

    Mapped to real controller indices (0-10)
    depending on direction (cw / ccw).
    """

    def __init__(self):
        self.tick = 0

        # durations per logical phase
        self.base_durations = np.array([25, 110, 310, 360, 260, 210, 210])
        self.delta = np.array([15, 20, 50, 30, 30, 30, 30])

        self.current_durations = None
        self.current_tick = None

        self.direction = None  # "cw" or "ccw"

        self.gen_new_ticks()

    def gen_new_ticks(self):
        self.tick = 0

        # --- Randomize durations ---
        self.current_durations = (self.base_durations + np.random.uniform(-self.delta, self.delta))

        self.current_durations = np.maximum(self.current_durations, 1)
        self.current_durations = np.round(self.current_durations).astype(int)

        self.current_tick = np.cumsum(self.current_durations)

        # --- Randomize direction per episode ---
        self.direction = np.random.choice(["cw", "ccw"])

    def _map_action(self, logical_idx):
        """
        Map logical phase index (0-6)
        to real controller index (0-10).
        """
        if self.direction == "cw":
            mapping = {
                0: 0,  # Hold
                1: 1,  # GoSafe
                2: 2,  # PrepareCW
                3: 3,  # RollCW
                4: 4,  # LandingCW
                5: 5,  # ProneCW
                6: 6  # StandUp
            }
        else:  # ccw
            mapping = {
                0: 0,  # Hold
                1: 1,  # GoSafe
                2: 7,  # PrepareCCW
                3: 8,  # RollCCW
                4: 9,  # LandingCCW
                5: 10,  # ProneCCW
                6: 6  # StandUp 
            }

        return mapping[logical_idx]

    def action(self):
        self.tick += 1

        idx = np.searchsorted(self.current_tick, self.tick)

        if idx < len(self.current_tick):
            return self._map_action(idx)

        return 0


def random_start():

    noise_q0 = np.random.uniform(-0.5, 0.5, (12, 1))
    q0 = q0_base.reshape(12, 1) + noise_q0

    yaw = random.uniform(-np.pi, np.pi)

    rpy = np.array([np.pi, 0, yaw])

    x0 = random.uniform(-1, 2.75)

    b0 = np.array([x0, 0, 0.3])

    return q0, rpy, b0


def save_data(data, episode_id, directory, direction, current_durations):

    save_path = os.path.join(directory, f"episode_{episode_id:04d}.npz")

    np.savez(save_path,
             com_pos=np.array(data["com_pos"]),
             com_vel=np.array(data["com_vel"]),
             ang_vel=np.array(data["ang_vel"]),
             rpy=np.array(data["rpy"]),
             q_pos=np.array(data["q_pos"]),
             q_vel=np.array(data["q_vel"]),
             controller=np.array(data["controller"]),
             direction=direction,
             durations=current_durations)

    print(f"Saved {save_path}")


def ask_to_save():

    while True:

        user_input = input("Save episode? [y/n/q]: ").lower()

        if user_input in ["y", "yes"]:
            return True

        elif user_input in ["n", "no"]:
            return False

        elif user_input == "q":
            exit()

        else:
            print("Please type y, n or q.")


def main():
    os.makedirs(directory_save, exist_ok=True)
    strategy_name = "tb"
    robot_sim = Go2ModelSimMuJoCo(render=render_sim, strategy_name=strategy_name)

    scheduler = TimeSchedulling()
    ep = 0
    while ep <= ep_num:

        print(f"\nEpisode {ep}")

        q0, rpy, b0 = random_start()

        robot_sim.reset_robot_pose(b0=b0, q0=q0, r0=rpy)

        scheduler.gen_new_ticks()

        total_duration = scheduler.current_tick[-1]

        episode_data = {
            "com_pos": [],
            "com_vel": [],
            "ang_vel": [],
            "rpy": [],
            "q_pos": [],
            "q_vel": [],
            "controller": []
        }

        for _ in range(total_duration):

            action = scheduler.action()

            robot_sim.control_loop(mode=action)

            state = robot_sim.robot_states

            episode_data["com_pos"].append(state.r_pos.copy())
            episode_data["com_vel"].append(state.r_vel.copy())
            episode_data["ang_vel"].append(state.omega.copy())
            episode_data["rpy"].append(state.rpy.copy())
            episode_data["q_pos"].append(state.q.copy())
            episode_data["q_vel"].append(state.dq.copy())
            episode_data["controller"].append(action)

        # ---------- Decide if saving ----------

        if render_sim:

            save = ask_to_save()

            if save:
                save_data(episode_data, ep, directory_save, scheduler.direction, scheduler.current_durations)
                ep += 1

        else:

            save_data(episode_data, ep, directory_save, scheduler.direction, scheduler.current_durations)
            ep += 1
        print(f"Saving dataset to: {directory_save}")


if __name__ == "__main__":
    main()
