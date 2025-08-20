#include "quantum_planner_server/ClientNode.hpp"

namespace quantum_planner_server
{

ClientNode::ClientNode()
: Node("quantum_client")
{
    // Declare parameters with default values
    // Goal parameters
    this->declare_parameter("goal.header.frame_id", "map");
    this->declare_parameter("goal.pose.position.x", 0.0);
    this->declare_parameter("goal.pose.position.y", 0.0);
    this->declare_parameter("goal.pose.position.z", 0.0);
    this->declare_parameter("goal.pose.orientation.x", 0.0);
    this->declare_parameter("goal.pose.orientation.y", 0.0);
    this->declare_parameter("goal.pose.orientation.z", 0.0);
    this->declare_parameter("goal.pose.orientation.w", 1.0);

    // Start parameters
    this->declare_parameter("start.header.frame_id", "map");
    this->declare_parameter("start.pose.position.x", 0.0);
    this->declare_parameter("start.pose.position.y", 0.0);
    this->declare_parameter("start.pose.position.z", 0.0);
    this->declare_parameter("start.pose.orientation.x", 0.0);
    this->declare_parameter("start.pose.orientation.y", 0.0);
    this->declare_parameter("start.pose.orientation.z", 0.0);
    this->declare_parameter("start.pose.orientation.w", 1.0);

    this->declare_parameter("max_planning_time_steps", 100);
    this->declare_parameter("planning_timeout", 10.0f);
    this->declare_parameter("robot_id", "Angie");

    // Planner 
    this->declare_parameter("planner.solver", "qaoa");
    this->declare_parameter("planner.fallback_to_classical", true);
    this->declare_parameter("planner.classical_backup", "dijkstra");

    // Planner params
    this->declare_parameter("planner.params.num_layers", 1);
    this->declare_parameter("planner.params.optimization_steps", 50);
    this->declare_parameter("planner.params.convergence_threshold", 1e-4f);
    this->declare_parameter("planner.params.backend", "qasm_simulator");
    this->declare_parameter("planner.params.shots", 1024);
    this->declare_parameter("planner.params.use_noise_model", false);
    this->declare_parameter("planner.params.optimizer", "SPSA");
    this->declare_parameter("planner.params.timeout_seconds", 30.0f);
    this->declare_parameter("planner.params.enable_classical_comparison", false);

    // Initialize the action client
    action_client_ = rclcpp_action::create_client<QuantumPath>(
        this,
        "compute_quantum_path" // action server name
    );

    RCLCPP_INFO(this->get_logger(), "Client has been started");

}

void ClientNode::send_request(QuantumPath::Goal goal)
{
    finished_ = false;
    success_ = false;

    if(!action_client_->wait_for_action_server()) {
        RCLCPP_ERROR(this->get_logger(), "Action server not available after waiting");
        return;
    }

    auto send_goal_options = rclcpp_action::Client<QuantumPath>::SendGoalOptions();

    send_goal_options.goal_response_callback =
        std::bind(&ClientNode::goal_response_callback, this, std::placeholders::_1);
    send_goal_options.feedback_callback =
        std::bind(&ClientNode::feedback_callback, this, std::placeholders::_1, std::placeholders::_2);
    send_goal_options.result_callback =
        std::bind(&ClientNode::result_callback, this, std::placeholders::_1);

    action_client_->async_send_goal(goal, send_goal_options);
    RCLCPP_INFO(this->get_logger(), "Goal sent to action server");
}

void ClientNode::goal_response_callback(
    rclcpp_action::ClientGoalHandle<QuantumPath>::SharedPtr goal_handle)
{
    if (!goal_handle) {
        RCLCPP_ERROR(this->get_logger(), "Goal was rejected by the action server");
        return;
    }
    RCLCPP_INFO(this->get_logger(), "Goal accepted by the action server");
}

void ClientNode::feedback_callback(
    rclcpp_action::ClientGoalHandle<QuantumPath>::SharedPtr,
    const std::shared_ptr<const QuantumPath::Feedback> feedback)
{
    RCLCPP_INFO(this->get_logger(), "Received feedback: %f%% complete", feedback->percent_complete);

}

void ClientNode::result_callback(const rclcpp_action::ClientGoalHandle<QuantumPath>::WrappedResult & result)
{
    finished_ = true;
    switch (result.code) {
        case rclcpp_action::ResultCode::SUCCEEDED:
            success_ = true;
            RCLCPP_INFO(this->get_logger(), "Goal succeeded");
            break;
        case rclcpp_action::ResultCode::ABORTED:
            RCLCPP_ERROR(this->get_logger(), "Goal was aborted");
            return;
        case rclcpp_action::ResultCode::CANCELED:
            RCLCPP_ERROR(this->get_logger(), "Goal was canceled");
            return;
        default:
            RCLCPP_ERROR(this->get_logger(), "Unknown result code");
            return;
    }
    // Process the result
    // For example, print the path or other relevant information
    RCLCPP_INFO(this->get_logger(), "Result received");
}

ClientNode::QuantumPath::Goal ClientNode::get_message(){
    auto msg = QuantumPath::Goal();
    
    // Fill in the message fields from parameters
    msg.goal.header.frame_id = this->get_parameter("goal.header.frame_id").as_string();
    msg.goal.pose.position.x = this->get_parameter("goal.pose.position.x").as_double();
    msg.goal.pose.position.y = this->get_parameter("goal.pose.position.y").as_double();
    msg.goal.pose.position.z = this->get_parameter("goal.pose.position.z").as_double();
    msg.goal.pose.orientation.x = this->get_parameter("goal.pose.orientation.x").as_double();
    msg.goal.pose.orientation.y = this->get_parameter("goal.pose.orientation.y").as_double();
    msg.goal.pose.orientation.z = this->get_parameter("goal.pose.orientation.z").as_double();
    msg.goal.pose.orientation.w = this->get_parameter("goal.pose.orientation.w").as_double();
    msg.goal.header.stamp = this->now();

    msg.start.header.frame_id = this->get_parameter("start.header.frame_id").as_string();
    msg.start.pose.position.x = this->get_parameter("start.pose.position.x").as_double();
    msg.start.pose.position.y = this->get_parameter("start.pose.position.y").as_double();
    msg.start.pose.position.z = this->get_parameter("start.pose.position.z").as_double();
    msg.start.pose.orientation.x = this->get_parameter("start.pose.orientation.x").as_double();
    msg.start.pose.orientation.y = this->get_parameter("start.pose.orientation.y").as_double();
    msg.start.pose.orientation.z = this->get_parameter("start.pose.orientation.z").as_double();
    msg.start.pose.orientation.w = this->get_parameter("start.pose.orientation.w").as_double();
    msg.start.header.stamp = this->now();

    msg.max_planning_time_steps = this->get_parameter("max_planning_time_steps").as_int();
    msg.planning_timeout = this->get_parameter("planning_timeout").as_double();
    msg.robot_id = this->get_parameter("robot_id").as_string();

    msg.planner.solver = this->get_parameter("planner.solver").as_string();
    msg.planner.fallback_to_classical = this->get_parameter("planner.fallback_to_classical").as_bool();
    msg.planner.classical_backup = this->get_parameter("planner.classical_backup").as_string();

    msg.planner.params.num_layers = this->get_parameter("planner.params.num_layers").as_int();
    msg.planner.params.optimization_steps = this->get_parameter("planner.params.optimization_steps").as_int();
    msg.planner.params.convergence_threshold = this->get_parameter("planner.params.convergence_threshold").as_double();
    msg.planner.params.backend = this->get_parameter("planner.params.backend").as_string();
    msg.planner.params.shots = this->get_parameter("planner.params.shots").as_int();
    msg.planner.params.use_noise_model = this->get_parameter("planner.params.use_noise_model").as_bool();
    msg.planner.params.optimizer = this->get_parameter("planner.params.optimizer").as_string();
    msg.planner.params.timeout_seconds = this->get_parameter("planner.params.timeout_seconds").as_double();
    msg.planner.params.enable_classical_comparison = this->get_parameter("planner.params.enable_classical_comparison").as_bool();
    
    return msg;
}


} // namespace quantum_planner_server
