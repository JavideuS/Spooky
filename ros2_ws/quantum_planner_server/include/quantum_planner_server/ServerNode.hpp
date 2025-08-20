#ifndef QUANTUM_PLANNER_SERVER__SERVERNODE_HPP_
#define QUANTUM_PLANNER_SERVER__SERVERNODE_HPP_

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <quantum_robotics_interface/action/compute_quantum_path.hpp>
#include <grid_map_msgs/msg/grid_map.hpp>

namespace quantum_planner_server {

class ServerNode : public rclcpp::Node
{
public:

    using QuantumPath = quantum_robotics_interface::action::ComputeQuantumPath;
    using GoalHandleQuantumPath = rclcpp_action::ServerGoalHandle<QuantumPath>;

    ServerNode();
    void callFastAPI();
    
private:
    rclcpp_action::Server<QuantumPath>::SharedPtr action_server_;

    rclcpp_action::GoalResponse handle_goal(
        const rclcpp_action::GoalUUID &,
        std::shared_ptr<const QuantumPath::Goal>);
    rclcpp_action::CancelResponse handle_cancel(std::shared_ptr<GoalHandleQuantumPath>);
    void handle_accepted(std::shared_ptr<GoalHandleQuantumPath>);
    
    void execute(std::shared_ptr<GoalHandleQuantumPath>);

    rclcpp::Subscription<grid_map_msgs::msg::GridMap>::SharedPtr map_sub_;

    grid_map_msgs::msg::GridMap map_;
    std::string map_path_;
    std::string materials_path_;

    std::string read_file(const std::string&); // To upload via API

};

} // namespace quantum_planner_server
#endif // QUANTUM_PLANNER_SERVER__SERVERNODE_HPP_