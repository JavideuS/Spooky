#include "quantum_planner_server/ServerNode.hpp"

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<quantum_planner_server::ServerNode>();
    
    // Call FastAPI endpoint
    // node->callFastAPI();
    
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}