import threading
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
 
 
class RobotActionController(Node):
    LINEAR_SPEED = 0.25   # м/с
    ANGULAR_SPEED = 0.6   # рад/с
    PUBLISH_RATE_HZ = 10.0
    CMD_VEL_TOPIC = "/cmd_vel"
 
    def __init__(self):
        super().__init__("piple_action_controller")
 
        self.action_to_twist = {
            "move_forward":  (self.LINEAR_SPEED, 0.0),
            "move_backward": (-self.LINEAR_SPEED, 0.0),
            "turn_left":     (0.0, self.ANGULAR_SPEED),
            "turn_right":    (0.0, -self.ANGULAR_SPEED),
            "stop":          (0.0, 0.0),
            "none":          (0.0, 0.0),
        }
 
        self._publisher = self.create_publisher(Twist, self.CMD_VEL_TOPIC, 10)
 
        self._lock = threading.Lock()
        self._linear = 0.0
        self._angular = 0.0
 
        self._timer = self.create_timer(
            1.0 / self.PUBLISH_RATE_HZ, self._publish_current_velocity
        )
 
        self.get_logger().info(
            f"Piple action controller запущен, публикую в {self.CMD_VEL_TOPIC}"
        )
 
    def execute_action(self, action: str) -> None:
        """Вызывается мозгом (piple_AI.py) с одним из ALLOWED_ACTIONS."""
        linear, angular = self.action_to_twist.get(action, (0.0, 0.0))
 
        with self._lock:
            self._linear = linear
            self._angular = angular
 
        self.get_logger().info(
            f"Action '{action}' -> linear={linear:.2f} angular={angular:.2f}"
        )
 
    def _publish_current_velocity(self) -> None:
        with self._lock:
            linear, angular = self._linear, self._angular
 
        msg = Twist()
        msg.linear.x = linear
        msg.angular.z = angular
        self._publisher.publish(msg)
 
    def shutdown(self) -> None:
        """Гарантируем, что робот не продолжит ехать после выхода из программы."""
        self.execute_action("stop")
        self._publish_current_velocity()
 
 
def start_ros_controller() -> "RobotActionController":
    """
    Инициализирует rclpy и запускает ноду в фоновом потоке, чтобы
    основной скрипт (piple_AI.py) мог продолжать работать со своим
    обычным input()-циклом.
    """
    rclpy.init()
    node = RobotActionController()
 
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()
 
    return node
 
 
def stop_ros_controller(node: "RobotActionController") -> None:
    node.shutdown()
    node.destroy_node()
    rclpy.shutdown()
 
