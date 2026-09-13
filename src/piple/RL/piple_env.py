import numpy as np
import mujoco
from gymnasium import spaces
from stable_baselines3.common.vec_env.base_vec_env import VecEnv
 
WHEEL_NAMES = ["front_left", "front_right", "rear_left", "rear_right"]
N_RANGE_RAYS = 8
MAX_LIN_VEL = 1.5      # m/s, tune to your robot
MAX_ANG_VEL = 3.0      # rad/s
TRACK_WIDTH = 0.38     # left-right wheel separation (matches URDF y-offsets *2)
WHEEL_RADIUS = 0.0795
MAX_EPISODE_STEPS = 500
CONTROL_DECIMATION = 4  # physics substeps per RL step (sim dt=0.005 -> control dt=0.02)
 
 
class PipleArenaEnv(VecEnv):
    def __init__(self, xml_path: str, n_robots: int):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.n_robots = n_robots
        self.episode_steps = np.zeros(n_robots, dtype=np.int32)
 
        # Cache ids up front -- looking things up by name every step is slow.
        self._joint_qpos_adr = []
        self._joint_qvel_adr = []
        self._wheel_actuator_ids = []
        self._sensor_adr = {"linvel": [], "angvel": [], "quat": []}
        self._range_sensor_adr = []
        self._init_qpos = []
 
        for i in range(n_robots):
            jid = self.model.joint(f"root_{i}").id
            self._joint_qpos_adr.append(self.model.jnt_qposadr[jid])
            self._joint_qvel_adr.append(self.model.jnt_dofadr[jid])
            self._init_qpos.append(self.data.qpos[
                self.model.jnt_qposadr[jid]:self.model.jnt_qposadr[jid] + 7
            ].copy())
 
            self._wheel_actuator_ids.append(
                [self.model.actuator(f"act_{w}_joint_{i}").id for w in WHEEL_NAMES]
            )
 
            self._sensor_adr["linvel"].append(self._sensor_slice(f"linvel_{i}"))
            self._sensor_adr["angvel"].append(self._sensor_slice(f"angvel_{i}"))
            self._sensor_adr["quat"].append(self._sensor_slice(f"quat_{i}"))
            self._range_sensor_adr.append(
                [self._sensor_slice(f"rf_range{r}_{i}") for r in range(N_RANGE_RAYS)]
            )
 
        obs_dim = 5 + N_RANGE_RAYS
        observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
        action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        super().__init__(n_robots, observation_space, action_space)
        self.render_mode = None
        self._actions = None
 
    def _sensor_slice(self, name):
        sid = self.model.sensor(name).id
        adr = self.model.sensor_adr[sid]
        dim = self.model.sensor_dim[sid]
        return slice(adr, adr + dim)
 
    # ---- VecEnv API --------------------------------------------------
    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        for i in range(self.n_robots):
            adr = self._joint_qpos_adr[i]
            self.data.qpos[adr:adr + 7] = self._init_qpos[i]
            # small random yaw so all 9 don't learn an identical fixed policy
            yaw = np.random.uniform(-np.pi, np.pi)
            self.data.qpos[adr + 3:adr + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        mujoco.mj_forward(self.model, self.data)
        self.episode_steps[:] = 0
        return self._get_obs()
 
    def step_async(self, actions):
        self._actions = np.clip(actions, -1.0, 1.0)
 
    def step_wait(self):
        v = self._actions[:, 0] * MAX_LIN_VEL
        w = self._actions[:, 1] * MAX_ANG_VEL
        v_left = (v - w * TRACK_WIDTH / 2.0) / WHEEL_RADIUS
        v_right = (v + w * TRACK_WIDTH / 2.0) / WHEEL_RADIUS
 
        for i in range(self.n_robots):
            fl, fr, rl, rr = self._wheel_actuator_ids[i]
            self.data.ctrl[fl] = v_left[i]
            self.data.ctrl[rl] = v_left[i]
            self.data.ctrl[fr] = v_right[i]
            self.data.ctrl[rr] = v_right[i]
 
        for _ in range(CONTROL_DECIMATION):
            mujoco.mj_step(self.model, self.data)
 
        obs = self._get_obs()
        rewards, dones, infos = self._get_reward_done_info()
        self.episode_steps += 1
 
        for i in np.where(dones)[0]:
            self._reset_one(i)
        if np.any(dones):
            obs = self._get_obs()  # re-read after any partial resets
 
        return obs, rewards, dones, infos
 
    def _reset_one(self, i):
        adr = self._joint_qpos_adr[i]
        vadr = self._joint_qvel_adr[i]
        self.data.qpos[adr:adr + 7] = self._init_qpos[i]
        yaw = np.random.uniform(-np.pi, np.pi)
        self.data.qpos[adr + 3:adr + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        self.data.qvel[vadr:vadr + 6] = 0.0
        self.episode_steps[i] = 0
 
    def _get_obs(self):
        obs = np.zeros((self.n_robots, self.observation_space.shape[0]), dtype=np.float32)
        for i in range(self.n_robots):
            linvel = self.data.sensordata[self._sensor_adr["linvel"][i]]
            angvel = self.data.sensordata[self._sensor_adr["angvel"][i]]
            quat = self.data.sensordata[self._sensor_adr["quat"][i]]  # w,x,y,z
            yaw = np.arctan2(
                2 * (quat[0] * quat[3] + quat[1] * quat[2]),
                1 - 2 * (quat[2] ** 2 + quat[3] ** 2),
            )
            ranges = np.array([
                self.data.sensordata[s][0] for s in self._range_sensor_adr[i]
            ])
            ranges = np.where(ranges < 0, 5.0, ranges)  # -1 = no hit -> treat as max range
            obs[i, 0:2] = linvel[0:2]
            obs[i, 2] = angvel[2]
            obs[i, 3] = np.sin(yaw)
            obs[i, 4] = np.cos(yaw)
            obs[i, 5:] = ranges
        return obs
 
    def _get_reward_done_info(self):
        rewards = np.zeros(self.n_robots, dtype=np.float32)
        dones = np.zeros(self.n_robots, dtype=bool)
        infos = [{} for _ in range(self.n_robots)]
        for i in range(self.n_robots):
            linvel = self.data.sensordata[self._sensor_adr["linvel"][i]]
            angvel = self.data.sensordata[self._sensor_adr["angvel"][i]]
            ranges = np.array([self.data.sensordata[s][0] for s in self._range_sensor_adr[i]])
            ranges = np.where(ranges < 0, 5.0, ranges)
            min_range = ranges.min()
 
            forward_speed = linvel[0]
            spin_penalty = 0.05 * abs(angvel[2])
            collision_penalty = 2.0 if min_range < 0.15 else 0.0
            energy_penalty = 0.01 * np.sum(self._actions[i] ** 2)
 
            rewards[i] = forward_speed - spin_penalty - collision_penalty - energy_penalty
 
            fell_over = self.data.qpos[self._joint_qpos_adr[i] + 2] < 0.02
            timeout = self.episode_steps[i] >= MAX_EPISODE_STEPS
            dones[i] = fell_over or timeout or (min_range < 0.15)
            if dones[i]:
                infos[i]["terminal_observation"] = None  # SB3 fills this in if needed
        return rewards, dones, infos
 
    # ---- boilerplate VecEnv requires but we don't need per-robot procs ----
    def close(self):
        pass
 
    def get_attr(self, attr_name, indices=None):
        return [getattr(self, attr_name)] * self.n_robots
 
    def set_attr(self, attr_name, value, indices=None):
        setattr(self, attr_name, value)
 
    def env_method(self, method_name, *args, indices=None, **kwargs):
        return [getattr(self, method_name)(*args, **kwargs)] * self.n_robots
 
    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.n_robots
 
