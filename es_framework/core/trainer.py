import time
import json
import numpy as np
import multiprocessing as mp

from tqdm import tqdm
from datetime import datetime
from pathlib import Path

from es_framework.workers.worker import init_worker, run_micro_task
from es_framework.logging.logger import TrainingLogger
from es_framework.models.checkpoint import ModelCheckpoint


# ----------------------------------------
# JSON SERIALIZATION
# ----------------------------------------
def make_json_serializable(obj):

    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj

    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [make_json_serializable(v) for v in obj]

    if isinstance(obj, type):
        return obj.__name__

    if hasattr(obj, "__dict__"):
        return {"__class__": obj.__class__.__name__, **{k: make_json_serializable(v) for k, v in obj.__dict__.items()}}
    return str(obj)


# ----------------------------------------
# TRAINER
# ----------------------------------------
class Trainer:

    def __init__(self, config):

        self.config = config

        self.pop_size = config["pop_size"]
        self.num_scenarios = config["num_scenarios"]
        self.max_generations = config["max_generations"]
        self.max_workers = config["max_workers"]

        self.env_config = config["env_config"]
        self.model_config = config["model_config"]
        self.env_spec = config["env_spec"]

        # ----------------------------------------
        # OPTIMIZER
        # ----------------------------------------
        self.optimizer = self._build_optimizer(config)

        # ----------------------------------------
        # SCENARIOS
        # ----------------------------------------
        self.scenarios = self._build_scenarios()

        # ----------------------------------------
        # RUN DIR
        # ----------------------------------------
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        task_name = self.config["job_name"]
        optimizer_name = self.config["optimizer_type"]

        self.run_dir = Path("experiments") / task_name / optimizer_name / timestamp
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # ----------------------------------------
        # LOGGER
        # ----------------------------------------
        self.logger = TrainingLogger(self.run_dir)

        # ----------------------------------------
        # SAVE CONFIG
        # ----------------------------------------
        config_path = self.run_dir / "config.json"

        config_to_save = make_json_serializable(self.config)

        with open(config_path, "w") as f:
            json.dump(config_to_save, f, indent=4)

        # ----------------------------------------
        # CHECKPOINT
        # ----------------------------------------
        self.checkpoint = ModelCheckpoint(self.run_dir)

        self.checkpoint.save_model_spec(env_spec=self.env_spec, model_cfg=self.model_config)

    # ----------------------------------------
    def _build_optimizer(self, config):

        if config["optimizer_type"] == "CEM":
            from es_framework.optimizers.cem import CEMOptimizer
            return CEMOptimizer(config)

        elif config["optimizer_type"] == "CMAES":
            from es_framework.optimizers.cma_es import CMAESOptimizer
            return CMAESOptimizer(config)

        else:
            raise NotImplementedError

    # ----------------------------------------
    def _build_scenarios(self):

        ScenarioClass = self.config["scenario_generator_class"]
        generator = ScenarioClass()

        return [generator.sample() for _ in range(self.num_scenarios)]

    # ----------------------------------------
    def _build_tasks(self, population):

        tasks = []
        task_id = 0

        for ind_id, params in enumerate(population):
            for scenario in self.scenarios:
                tasks.append((task_id, ind_id, params, scenario, self.config.get("difficulty", 0), None))
                task_id += 1

        return tasks

    # ----------------------------------------
    def _aggregate_fitness(self, results):

        fitness_dict = {i: [] for i in range(self.pop_size)}

        for _, ind_id, reward, _ in results:
            fitness_dict[ind_id].append(reward)

        return np.array([np.mean(fitness_dict[i]) if fitness_dict[i] else -1e6 for i in range(self.pop_size)])

    # ----------------------------------------
    def train(self):

        def create_pool(heartbeat):
            return mp.Pool(processes=self.max_workers, initializer=init_worker, initargs=(self.config, heartbeat))

        manager = mp.Manager()
        heartbeat = manager.dict()
        pool = create_pool(heartbeat)

        timeout_sec = 15.0
        startup_grace_sec = 5.0

        for gen in range(self.max_generations):

            print(f"\n[GEN {gen}]")

            heartbeat.clear()

            population = self.optimizer.sample()
            tasks = self._build_tasks(population)

            task_map = {t[0]: t for t in tasks}
            total_tasks = len(tasks)

            completed_results = {}

            gen_start = time.time()

            # ----------------------------------------
            # DISPATCH
            # ----------------------------------------
            futures = {pool.apply_async(run_micro_task, (task,)): task for task in tasks}

            pbar = tqdm(total=total_tasks, desc=f"GEN {gen}", leave=False)

            while len(completed_results) < total_tasks:

                restart_pool = False

                # ----------------------------------------
                # CHECK COMPLETED
                # ----------------------------------------
                done_futs = []

                for fut, task in list(futures.items()):

                    task_id = task[0]

                    if task_id in completed_results:
                        done_futs.append(fut)
                        continue

                    if fut.ready():

                        try:
                            result = fut.get()
                            completed_results[task_id] = result

                        except Exception as e:

                            print(f"[WARN] Task {task_id} failed: {e}")

                            _, ind_id, _, _, _, _ = task
                            completed_results[task_id] = (task_id, ind_id, -1e6, None)

                        pbar.update(1)
                        done_futs.append(fut)

                # Remove after iteration is complete
                for fut in done_futs:
                    futures.pop(fut, None)

                # ----------------------------------------
                # WATCHDOG
                # ----------------------------------------
                now = time.time()

                if now - gen_start > startup_grace_sec:

                    for task_id, last in list(heartbeat.items()):

                        if task_id in completed_results:
                            continue

                        if last <= 0:
                            continue

                        if now - last > timeout_sec:

                            print(f"[WATCHDOG] Task {task_id} timeout → restarting pool")

                            task = task_map[task_id]
                            _, ind_id, _, _, _, _ = task

                            # penalize
                            completed_results[task_id] = (task_id, ind_id, -1e6, None)

                            pbar.update(1)

                            # avoid re-trigger
                            heartbeat[task_id] = -1.0

                            # RESTART POOL
                            restart_pool = True
                            break

                if restart_pool:

                    # ----------------------------------------
                    # KILL POOL
                    # ----------------------------------------
                    pool.terminate()
                    pool.join()

                    # ----------------------------------------
                    # RECREATE POOL
                    # ----------------------------------------
                    manager = mp.Manager()
                    heartbeat = manager.dict()
                    pool = create_pool(heartbeat)

                    # ----------------------------------------
                    # REBUILD FUTURES (ONLY UNFINISHED TASKS)
                    # ----------------------------------------
                    futures = {}

                    for task in tasks:
                        task_id = task[0]

                        if task_id in completed_results:
                            continue

                        fut = pool.apply_async(run_micro_task, (task,))
                        futures[fut] = task

                    continue  # back to main loop

                time.sleep(0.2)

            pbar.close()

            # ----------------------------------------
            # GENERATION END
            # ----------------------------------------
            gen_time = time.time() - gen_start
            # mean_ind_time = gen_time / self.pop_size

            results = list(completed_results.values())
            fitness = self._aggregate_fitness(results)

            self.optimizer.update(fitness)

            # ----------------------------------------
            # CHECKPOINT
            # ----------------------------------------
            self.checkpoint.update(self.optimizer, gen)
            self.checkpoint.save_last(self.optimizer)
            self.checkpoint.save_periodic(self.optimizer, gen, interval=50)

            # ----------------------------------------
            # LOGGING
            # ----------------------------------------
            optimizer_metrics = self.optimizer.get_metrics()

            self.logger.log_generation(generation=gen,
                                       fitness=fitness,
                                       population=population,
                                       optimizer_metrics=optimizer_metrics,
                                       extra_metrics={"gen_time": gen_time})

        pool.close()
        pool.join()
        self.logger.close()
