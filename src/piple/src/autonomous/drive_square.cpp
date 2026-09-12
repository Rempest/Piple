#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <chrono>
int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<rclcpp::Node>("auto_drive");
    auto publisher =node->create_publisher<geometry_msgs::msg::Twist>("cmd_vel",10);
    int state = 0;
    auto start_time =std::chrono::steady_clock::now();
    auto timer = node->create_wall_timer(std::chrono::milliseconds(100),
       [publisher, &state, &start_time](){
        geometry_msgs::msg::Twist msg;
        auto elapsed = std::chrono::steady_clock::now() - start_time;
            if(state == 0)
            {
                msg.linear.x = 1.0;
                msg.angular.z = 0.0;
                if(elapsed >= std::chrono::milliseconds(2500))
                {
                    state = 1;
                    start_time = std::chrono::steady_clock::now();
                }
            }
            else if(state == 1)
            {
                msg.linear.x = 0.0;
                msg.angular.z = 3.14; 
            if(elapsed >= std::chrono::milliseconds(1500))
                {
                    state = 0;
                    start_time = std::chrono::steady_clock::now();
                }
            }
            publisher->publish(msg);
        });
    rclcpp::spin(node);
    rclcpp::shutdown();
}