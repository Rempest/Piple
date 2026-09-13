import mujoco
import mujoco.viewer
from stable_baselines3.common.callbacks import BaseCallback
 
 
class LiveViewerCallback(BaseCallback):
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._viewer = None
 
    def _on_training_start(self) -> None:
        env = self.training_env
        self._viewer = mujoco.viewer.launch_passive(env.model, env.data)
 
    def _on_step(self) -> bool:
        if self._viewer is not None:
            if not self._viewer.is_running():
                return True
            self._viewer.sync()
        return True
 
    def _on_training_end(self) -> None:
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
 
