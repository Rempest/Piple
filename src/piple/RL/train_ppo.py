from stable_baselines3 import PPO
from piple_env import PipleArenaEnv
 
XML_PATH = "piple_arena.xml"   # adjust to wherever it sits relative to this script
N_ROBOTS = 9
 
env = PipleArenaEnv(XML_PATH, N_ROBOTS)
 
model = PPO(
    "MlpPolicy",
    env,
    n_steps=256,          # rollout length per robot before each update
    batch_size=256,       # 256 * 9 robots = 2304 samples per update
    learning_rate=3e-4,
    policy_kwargs=dict(net_arch=[64, 64]),  # small net -- no GPU, keep it light
    verbose=1,
    tensorboard_log="./tb_logs",
)
 
model.learn(total_timesteps=2_000_000)
model.save("piple_ppo")
