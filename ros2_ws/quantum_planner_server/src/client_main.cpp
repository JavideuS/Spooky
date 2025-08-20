#include "quantum_planner_server/ClientNode.hpp"

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<quantum_planner_server::ClientNode>();
    
    std::cout << "Waiting for action server to finish..." << std::endl;
    auto goal = node->get_message();
    node->send_request(goal);

    std::cout << "Waiting for action server to finish..." << std::endl;
    // rclcpp::Rate rate(10);
    // while (rclcpp::ok() && !node->is_action_finished()) {
    //     rclcpp::spin_some(node);
    //     rate.sleep();
    // }

    // if (node->is_result_success()) {
    //     std::cout << "Result: Success" << std::endl;
    // } else {
    //     std::cerr << "Result: error" << std::endl;
    // }

    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}