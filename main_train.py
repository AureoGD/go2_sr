import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import time
import torch
import numpy as np
import multiprocessing as mp
from tqdm import tqdm

from es_framework.optimizers import get_optimizer
from environment.learning_phases import LearningPhases
from es_framework.components.curriculum_manager import CurriculumManager
from es_framework.components.worker import init_worker, run_micro_task
from es_framework.components.policy import Policy
from es_framework.components.logger import TrainingLogger
from environment.go2_env import Go2Env


# ============================================================
# POOL CREATION
# ============================================================
def create_pool(num_workers, config, heartbeat_array):
    return mp.Pool(processes=num_workers, initializer=init_worker, initargs=(config, heartbeat_array))


# ============================================================
# MAIN
# ============================================================
def main():

    # --------------------------------------------------------
    # CONFIG
    # --------------------------------------------------------
    config = {
        "optimizer_type": "CEM",
        "job_name": "go2_self_righting",
        "pop_size": 40,
        "num_scenarios": 5,
        "max_generations": 250,
        "max_workers": 20,
        "sigma_init": 0.05,
        "sigma_decay": 0.995,
        "elite_frac": 0.2,
        "model_config": {
            "hidden_dims": [128, 128],
            "activation": "tanh",
            "discrete": True
        }
    }

    # --------------------------------------------------------
    # ENV SHAPES
    # --------------------------------------------------------
    dummy_env = Go2Env(rendering=False)
    obs_dim = dummy_env.observation_space.shape[0]
    act_dim = dummy_env.action_space.n
    dummy_env.close()

    policy = Policy(obs_dim, act_dim, **config["model_config"])

    params = list(policy.parameters())
    assert len(params) > 0, "Policy has no parameters!"

    phases = LearningPhases()
    curriculum = CurriculumManager(phases=phases)
    optimizer = get_optimizer(config["optimizer_type"], config, policy)

    logger = TrainingLogger(config=config, root_dir="results")
    logger.set_reference_model(policy)

    num_workers = config["max_workers"]
    total_tasks = config["pop_size"] * config["num_scenarios"]

    # --------------------------------------------------------
    # HEARTBEAT (PER TASK)
    # --------------------------------------------------------
    heartbeat_array = mp.Array("d", total_tasks)

    pool = create_pool(num_workers, config, heartbeat_array)

    STARTUP_TIME = time.time()
    STARTUP_GRACE_SEC = 30.0

    global_stats = None
    gen = 1

    print(f"[Main] Pool started with {num_workers} workers")

    # --------------------------------------------------------
    # TRAIN LOOP
    # --------------------------------------------------------
    try:
        while gen < config["max_generations"] + 1:
            gen_start = time.time()

            candidates = optimizer.ask()
            scenarios = curriculum.get_reset_conditions(config["num_scenarios"])
            difficulty = curriculum.get_difficulty()

            # --------------------------------------------
            # BUILD TASKS
            # --------------------------------------------
            tasks = []
            task_id = 0
            for i in range(config["pop_size"]):
                for s in scenarios:
                    tasks.append((task_id, candidates[i], s, difficulty, global_stats))
                    task_id += 1

            # --------------------------------------------
            # RESET HEARTBEATS
            # --------------------------------------------
            for i in range(total_tasks):
                heartbeat_array[i] = 0.0

            # --------------------------------------------
            # DISPATCH
            # --------------------------------------------
            futures = [pool.apply_async(run_micro_task, (t,)) for t in tasks]
            results = []

            pbar = tqdm(total=len(futures), desc=f"Gen {gen:03d}", leave=False)

            # --------------------------------------------
            # MONITOR
            # --------------------------------------------
            while futures:
                remaining = []

                for fut in futures:
                    if fut.ready():
                        try:
                            results.append(fut.get())
                        except Exception as e:
                            print(f"[Main] Task exception: {e}", flush=True)
                        pbar.update(1)
                    else:
                        remaining.append(fut)

                futures = remaining

                now = time.time()
                if now - STARTUP_TIME > STARTUP_GRACE_SEC:
                    for tid in range(total_tasks):
                        last = heartbeat_array[tid]
                        if last <= 0.0:
                            continue
                        if now - last > 15.0:
                            raise RuntimeError(f"Task {tid} heartbeat timeout")

                time.sleep(0.5)

            pbar.close()

            # --------------------------------------------
            # PROCESS RESULTS
            # --------------------------------------------
            rewards = {i: [] for i in range(config["pop_size"])}
            worker_stats = []

            for tid, r, stats, _ in results:
                rewards[tid // config["num_scenarios"]].append(r)
                if stats is not None:
                    worker_stats.append(stats)

            rewards_np = np.array([np.mean(rewards[i]) for i in range(config["pop_size"])])

            optimizer.tell(candidates, rewards_np)

            if worker_stats:
                global_stats = worker_stats[0]

            curriculum.update(success_rate=float(np.mean(rewards_np > 0.0)), optimizer=optimizer)

            gen_time = time.time() - gen_start
            print(f"Gen {gen:03d} | "
                  f"AvgR: {np.mean(rewards_np):7.2f} | "
                  f"Sigma: {optimizer.sigma:.4f} | "
                  f"Time: {gen_time:5.1f}s")

            logger.log_generation(gen, list(zip(candidates, rewards_np)), optimizer, global_stats,
                                  {"gen_time": gen_time})

            gen += 1

    # --------------------------------------------------------
    # WATCHDOG RECOVERY
    # --------------------------------------------------------
    except RuntimeError as e:
        print(f"\n[WATCHDOG] {e}")
        print("[WATCHDOG] Restarting pool")

        pool.terminate()
        pool.join()

        heartbeat_array = mp.Array("d", total_tasks)
        pool = create_pool(num_workers, config, heartbeat_array)

    except KeyboardInterrupt:
        print("\n[Main] Interrupted")

    finally:
        if pool:
            pool.close()
            pool.join()
        logger.close()


# ============================================================
# ENTRY
# ============================================================
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    print(f"[PID] Main: {os.getpid()}")
    main()
