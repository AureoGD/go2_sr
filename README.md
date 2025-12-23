# Go2 Self-Righting (Evolutionary Strategies)

This project implements a learning framework for the Unitree Go2 robot to master self-righting maneuvers using Evolutionary Strategies (CEM/CMA-ES). The project architecture is stratified into three primary modules: the **Training Framework**, the **Simulation Environment**, and **Utilities**.

## Project Structure

```text
go2_sr/
├── main_train.py                # ENTRY POINT: Launches the training loop
├── main_test.py                 # ENTRY POINT: Evaluates a trained policy
│
├── es_framework/                # === TRAINING FRAMEWORK ===
│   │                              (Optimization and Rollout Logic)
│   ├── optimizers/              # Algorithms (CEM, CMA-ES) to update weights
│   └── components/
│       ├── worker.py            # Parallel rollout manager
│       ├── policy.py            # Neural Network architecture
│       ├── curriculum_manager.py# Logic for increasing difficulty levels
│       └── nn_utils.py          # Helper functions for parameter handling
│
├── environment/                 # === SIMULATION ENVIRONMENT ===
│   │                              (Physics, Robot, and Task Definitions)
│   ├── go2_env.py               # Gym-like environment wrapper
│   ├── go2_sim.py               # Core simulation logic (MuJoCo interface)
│   ├── strategies/              # Self-righting task controllers
│   │   ├── unitree_self_righting.py # Unitree's default open-loop solution
│   │   ├── scheduler_rgc_mpc.py     # Multi-modal RGC-MPC based self-righting
│   │   └── rgc_mpc/                 # RGC-MPC controller strategy phases
│   └── assets/                  # Robot URDFs, MJCFs, and terrain tools
│
├── utilities/                   # === UTILITIES ===
│   └── inspect_metrics.py       # Diagnostic scripts
│
└── analysis/                    # Placeholder for post-training graphs/videos