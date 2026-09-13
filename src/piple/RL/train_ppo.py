from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
 
from piple_env import PipleArenaEnv
 
 
XML_PATH = "piple_arena.xml"
N_ROBOTS = 9
 
 
env = PipleArenaEnv(
    XML_PATH,
    N_ROBOTS,
)
 
 
model = PPO(
    "MlpPolicy",
    env,
    n_steps=256,
    batch_size=256,
    learning_rate=3e-4,
    policy_kwargs=dict(
        net_arch=[64, 64],
    ),
    verbose=1,
    tensorboard_log="./tb_logs",
)

checkpoint_callback = CheckpointCallback(
    save_freq=10_000,
    save_path="./checkpoints",
    name_prefix="piple_ppo",
    save_replay_buffer=False,
    save_vecnormalize=False,
)
 
 
model.learn(
    total_timesteps=2_000_000,
    callback=checkpoint_callback,
)
 
 
model.save("piple_ppo_final")
 
