import argparse
import glob
import os
import time
 
import mujoco
import mujoco.viewer
from stable_baselines3 import PPO
 
from piple_env import PipleArenaEnv
 
XML_PATH = "piple_arena.xml"
N_ROBOTS = 9
CONTROL_DT = 0.02
POLL_INTERVAL = 2.0  
 
 
def find_latest_checkpoint(checkpoint_dir: str, pattern: str):
    files = glob.glob(os.path.join(checkpoint_dir, pattern))
    if not files:
        return None
    return max(files, key=os.path.getmtime)
 
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", default="./checkpoints")
    parser.add_argument("--pattern", default="piple_ppo_*_steps.zip")
    args = parser.parse_args()
 
    env = PipleArenaEnv(XML_PATH, N_ROBOTS)
    obs = env.reset()
 
    model = None
    loaded_path = None
    last_poll = 0.0
 
    print(f"Watching {args.checkpoint_dir}/{args.pattern} for checkpoints...")
    print("Waiting for the first checkpoint to appear (this can take a "
          "minute after training starts, depending on save_freq)...")
 
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            step_start = time.time()
 
            # Periodically check for a newer checkpoint on disk.
            if step_start - last_poll > POLL_INTERVAL:
                last_poll = step_start
                latest = find_latest_checkpoint(args.checkpoint_dir, args.pattern)
                if latest is not None and latest != loaded_path:
                    try:
                        model = PPO.load(latest)
                        loaded_path = latest
                        print(f"Loaded checkpoint: {latest}")
                    except Exception as e:
                        # file may still be mid-write; just try again next poll
                        print(f"Could not load {latest} yet ({e}), will retry")
 
            if model is not None:
                actions, _ = model.predict(obs, deterministic=True)
                env.step_async(actions)
                obs, rewards, dones, infos = env.step_wait()
            # if no checkpoint loaded yet, just let physics sit idle (ctrl=0)
            else:
                mujoco.mj_step(env.model, env.data)
 
            viewer.sync()
 
            elapsed = time.time() - step_start
            sleep_time = CONTROL_DT - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
 
 
if __name__ == "__main__":
    main()
 
