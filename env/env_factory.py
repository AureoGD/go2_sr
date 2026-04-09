import mujoco
import mujoco.viewer
import gymnasium as gym

from es_framework.core.env_spec import EnvSpec
from env.normalizer import StateNormalizer
from tpe.core.tpe_module import TPEModule


def create_env(env_config):

    # -----------------------------
    # Mujoco
    # -----------------------------
    model = mujoco.MjModel.from_xml_path(env_config.scene_path)
    data = mujoco.MjData(model)

    viewer = None
    if env_config.render:
        try:
            viewer = mujoco.viewer.launch_passive(model, data)
        except Exception as e:
            print(f"[WARN] Viewer not available: {e}")
            viewer = None

    # -----------------------------
    # Components
    # -----------------------------
    controller = env_config.controller_class()

    normalizer = StateNormalizer(**env_config.normalizer_params)

    tpe = TPEModule(model_path=env_config.tpe_model_path)

    task = env_config.task_class(normalizer=normalizer, tpe=tpe)

    # -----------------------------
    # Env
    # -----------------------------
    env = env_config.env_class(urdf_path=env_config.urdf_path,
                               mj_model=model,
                               mj_data=data,
                               controller=controller,
                               task=task,
                               viewer=viewer)

    # -----------------------------
    # Build EnvSpec
    # -----------------------------
    obs_dim = env.observation_space.shape[0]

    if isinstance(env.action_space, gym.spaces.Discrete):
        act_dim = env.action_space.n
        is_discrete = True

    elif isinstance(env.action_space, gym.spaces.Box):
        act_dim = env.action_space.shape[0]
        is_discrete = False

    else:
        raise NotImplementedError(f"Unsupported action space: {type(env.action_space)}")

    env_spec = EnvSpec(obs_dim=obs_dim, act_dim=act_dim, is_discrete=is_discrete)

    return env, env_spec
