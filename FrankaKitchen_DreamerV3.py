try:
    import gymnasium_robotics  # noqa
except (ImportError, ModuleNotFoundError):
    print("You have to `pip install gymnasium_robotics` in order to run this example!")

import gymnasium as gym
from ray.rllib.algorithms.dreamerv3.dreamerv3 import DreamerV3Config
from ray import tune

# Set number of GPUs to use
num_gpus = 2

# Register the kitchen environment with the specific microwave task
def kitchen_microwave_env_creator(ctx):
    return gym.make("FrankaKitchen-v1", tasks_to_complete=["microwave"])

tune.register_env("kitchen-microwave", kitchen_microwave_env_creator)

# Configure DreamerV3
config = DreamerV3Config()
w = config.world_model_lr
c = config.critic_lr

config = (
    DreamerV3Config()
    .framework("torch")  # ✅ THIS FIXES THE ERROR
    .resources(
        num_cpus_for_main_process=8 * (num_gpus or 1),
    )
    .learners(
        num_learners=0 if num_gpus == 1 else num_gpus,
        num_gpus_per_learner=1 if num_gpus else 0,
    )
    .env_runners(num_envs_per_env_runner=8 * (num_gpus or 1), remote_worker_envs=True)
    .reporting(
        metrics_num_episodes_for_smoothing=(num_gpus or 1),
        report_images_and_videos=False,
        report_dream_data=False,
        report_individual_batch_item_stats=False,
    )
    .training(
        model_size="L",
        training_ratio=64,
        batch_size_B=16 * (num_gpus or 1),
        world_model_lr=[[0, 0.4 * w], [50000, 0.4 * w], [100000, 3 * w]],
        critic_lr=[[0, 0.4 * c], [50000, 0.4 * c], [100000, 3 * c]],
        actor_lr=[[0, 0.4 * c], [50000, 0.4 * c], [100000, 3 * c]],
    )
    .environment(env="kitchen-microwave")
)

# Run training
tune.Tuner(
    "DreamerV3",
    param_space=config.to_dict(),
    run_config=tune.RunConfig(
        stop={"timesteps_total": 1_000_000},
        name="dreamerv3_kitchen_microwave"
    ),
).fit()
