#ifndef QUANTUM_PLANNER_SERVER__CLIENTNODE_HPP_
#define QUANTUM_PLANNER_SERVER__CLIENTNODE_HPP_

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <quantum_robotics_interface/action/compute_quantum_path.hpp>

namespace quantum_planner_server
{

class ClientNode : public rclcpp::Node
{
public:
    using QuantumPath = quantum_robotics_interface::action::ComputeQuantumPath;
    using GoalHandleQuantumPath = rclcpp_action::ServerGoalHandle<QuantumPath>;

    ClientNode();
    void send_request(QuantumPath::Goal goal);
    QuantumPath::Goal get_message();

    bool is_action_finished() {return finished_;}
    bool is_result_success() {return success_;}

protected:
    virtual void goal_response_callback(rclcpp_action::ClientGoalHandle<QuantumPath>::SharedPtr goal_handle);
    virtual void feedback_callback(
        rclcpp_action::ClientGoalHandle<QuantumPath>::SharedPtr,
        const std::shared_ptr<const QuantumPath::Feedback>);
    virtual void result_callback(const rclcpp_action::ClientGoalHandle<QuantumPath>::WrappedResult & result);

private:
    rclcpp_action::Client<QuantumPath>::SharedPtr action_client_;


    bool finished_ {false};
    bool success_ {false};

};


} // namespace quantum_planner_server

#endif // QUANTUM_PLANNER_SERVER__CLIENTNODE_HPP_