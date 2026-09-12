#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
int main(int argc, char **argv){
    rclcpp::init(argc, argv);
    auto node = std::make_shared<rclcpp::Node>("twist_adapter");
    auto publisher = node->create_publisher<geometry_msgs::msg::TwistStamped>("/diff_drive_controller/cmd_vel", 10);
    auto subscription = node->create_subscription<geometry_msgs::msg::Twist>("/cmd_vel", 10, [node, publisher](const geometry_msgs::msg::Twist::SharedPtr msg){
        RCLCPP_INFO(
            node->get_logger(),
                             "Linear X: %.3f\n"
                             "Linear Y: %.3f\n"
                             "Linear Z: %.3f\n"
                             "Angular X: %.3f\n"
                             "Angular Y: %.3f\n"
                             "Angular Z: %.3f\n", 
                             msg->linear.x, 
                             msg->linear.y, 
                             msg->linear.z,
                             msg->angular.x, 
                             msg->angular.y, 
                             msg->angular.z
                            );
                        geometry_msgs::msg::TwistStamped stamped;
                    stamped.header.stamp = node->get_clock()->now();
                    stamped.header.frame_id = "base_link";
                    stamped.twist = *msg;
                    publisher->publish(stamped);});
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}