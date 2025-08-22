#include "quantum_planner_server/ServerNode.hpp"

#include <httplib.h>
#include <nlohmann/json.hpp>
#include "ament_index_cpp/get_package_share_directory.hpp"

#include <fstream>
#include <sstream>
#include <iostream>

namespace quantum_planner_server
{

using json = nlohmann::json;
using MultipartFormDataItems = std::vector<httplib::MultipartFormData>;


ServerNode::ServerNode()
: Node("quantum_server")
{
    // Initialize the action server
    action_server_ = rclcpp_action::create_server<QuantumPath>(
        this,
        "compute_quantum_path",
        std::bind(&ServerNode::handle_goal, this, std::placeholders::_1, std::placeholders::_2),
        std::bind(&ServerNode::handle_cancel, this, std::placeholders::_1),
        std::bind(&ServerNode::handle_accepted, this, std::placeholders::_1)
    );

    this->map_sub_ = this->create_subscription<grid_map_msgs::msg::GridMap>(
        "map",
        rclcpp::QoS(10),
        [this](const grid_map_msgs::msg::GridMap::SharedPtr msg) {
            this->map_ = *msg;
            RCLCPP_INFO(this->get_logger(), "Received grid map with %zu layers", msg->layers.size());
            
            RCLCPP_INFO(this->get_logger(), "Layer: %s", msg->layers[0].c_str());
            RCLCPP_INFO(this->get_logger(), "Map size: %f x %f", msg->info.length_x, msg->info.length_y);
        }
    );
    std::string package_path = ament_index_cpp::get_package_share_directory("quantum_planner_server");

    this->map_path_ = package_path + "/maps/no_obs3x3_mix.h5";
    this->materials_path_ = package_path + "/maps/materials.yaml";
    RCLCPP_INFO(this->get_logger(), "Map path set to: %s", this->map_path_.c_str());

    RCLCPP_INFO(this->get_logger(), "Server has been started");
}

void ServerNode::callFastAPI()
{
    // Create an HTTP client
    httplib::Client cli("http://localhost:8000"); // Replace with your FastAPI IP if remote

    // Define the POST body (e.g., JSON)
    std::string body = R"({"robot_id": "Angie", "template": "quantum-agent"})";

    // Perform the POST request
    auto res = cli.Post("/robots", body, "application/json");

    // Check if the request was successful
    if (res && res->status == 200) {
        std::cout << "POST successful!" << std::endl;
        std::cout << "Response: " << res->body << std::endl;
    } else {
        std::cout << "POST failed or no response" << std::endl;
        if (res) {
            std::cout << "Status: " << res->status << std::endl;
        } else {
            std::cout << "No response received (check connection/server)" << std::endl;
        }
    }

    // Prepare the request body
    res = cli.Get("/robots");

    if (res && res->status == 200) {
        RCLCPP_INFO(this->get_logger(), "Received: %s", res->body.c_str());
    }
}

rclcpp_action::GoalResponse ServerNode::handle_goal(
    const rclcpp_action::GoalUUID & uuid,
    std::shared_ptr<const QuantumPath::Goal> goal)
{
    (void)uuid; // Unused parameter
    auto start_point = goal->start.pose.position;
    auto goal_point = goal->goal.pose.position;
    auto x_max = this->map_.info.length_x;
    auto y_max = this->map_.info.length_y;

    if(this->map_.layers.empty()) {
        RCLCPP_ERROR(this->get_logger(), "No map data available, cannot process goal");
        return rclcpp_action::GoalResponse::REJECT;
    }
    else if(start_point.x < 0 || start_point.y < 0 ||
            goal_point.x < 0 || goal_point.y < 0 ||
            start_point.x > x_max || start_point.y > y_max ||
            goal_point.x > x_max || goal_point.y > y_max) {
        // Check if start and goal coordinates are within the map bounds
        RCLCPP_ERROR(this->get_logger(), "Start or goal coordinates are out of bounds");
        return rclcpp_action::GoalResponse::REJECT;
    }
    RCLCPP_INFO(this->get_logger(), "Received goal request");
    RCLCPP_INFO(this->get_logger(), "Start: (%f, %f), Goal: (%f, %f)",
        start_point.x, start_point.y, goal_point.x, goal_point.y);

    return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
    
}
rclcpp_action::CancelResponse ServerNode::handle_cancel(
    const std::shared_ptr<GoalHandleQuantumPath> goal_handle)
  {
    (void)goal_handle;
    RCLCPP_INFO(this->get_logger(), "Canceling goal");
    return rclcpp_action::CancelResponse::ACCEPT;
}

void ServerNode::handle_accepted(std::shared_ptr<GoalHandleQuantumPath> goal_handle)
{
    RCLCPP_INFO(this->get_logger(), "Goal accepted, executing...");
    std::thread(&ServerNode::execute, this, goal_handle).detach();  
    // Maybe should implement thread pool for swarms
}

void ServerNode::execute(std::shared_ptr<GoalHandleQuantumPath> goal_handle)
{    
    const auto goal = goal_handle->get_goal();
    auto feedback = std::make_shared<QuantumPath::Feedback>();
    auto result = std::make_shared<QuantumPath::Result>();
    
    // Checking the goal was not canceled
    if (goal_handle->is_canceling()) {
        // result->percent_complete = 0;
        // goal_handle->canceled(result);
        RCLCPP_INFO(this->get_logger(), "Goal canceled");
        goal_handle->canceled(result);
    }
    
    // Create an HTTP client
    httplib::Client cli("http://localhost:8000");

    json j;

    j["robot_id"] = goal->robot_id;
    j["template"] = "default";

    std::string body = j.dump();

    // Perform the POST request
    auto res = cli.Post("/robots", body, "application/json");

    // Check if the request was successful
    if (res && res->status == 200) {
        std::cout << "POST successful!" << std::endl;
        std::cout << "Response: " << res->body << std::endl;
    } else {
        std::cout << "POST failed or no response" << std::endl;
        if (res) {
            std::cout << "Status: " << res->status << std::endl;
        } else {
            std::cout << "No response received (check connection/server)" << std::endl;
        }
        goal_handle->abort(result);
    }

    // If map not loaded, we load it
    // And it's a good to check if map is fine
    std::string api_path = "/robots/" + goal->robot_id + "/maps/" + goal->map_id;

    res = cli.Get(api_path.c_str());
    if (res && res->status == 200) {
        std::cout << "Map already exists, skipping upload." << std::endl;
        std::cout << "Response: " << res->body << std::endl;
        // If map exists, we can skip uploading it
    } else {
        std::cout << "Map does not exist, uploading..." << std::endl;
        
        // Uploading map
        // This should happen only once per map and robot_id

        j.clear();

        std::string map_data = read_file(map_path_);
        std::string mat_data = read_file(materials_path_);

        httplib::UploadFormDataItems items = {
            {"file", map_data, "map.h5", "application/octet-stream"},
            {"materials_file", mat_data, "materials.yaml", "text/yaml"}
        };


        RCLCPP_INFO(this->get_logger(), "Map %s", this->map_path_.c_str());

        res = cli.Post(api_path.c_str(), items);

        // Check if the request was successful
        if (res && res->status == 200) {
            std::cout << "Map POST successful!" << std::endl;
            std::cout << "Response: " << res->body << std::endl;
        } else {
            std::cout << "Map POST failed or no response" << std::endl;
            if (res) {
                std::cout << "Status: " << res->status << std::endl;
            } else {
                std::cout << "No response received (check connection/server)" << std::endl;
            }
            goal_handle->abort(result);;
        }

    }

    // Finally after making we have a robot_id and map
    // We can call to solve the path planning problem
    j.clear();

    api_path = "/robots/" + goal->robot_id + "/plan";

    j["map_id"] = goal->map_id; // Use the map name instead of "default"
    j["start"] = {goal->start.pose.position.x, goal->start.pose.position.y};
    j["goal"] = {goal->goal.pose.position.x, goal->goal.pose.position.y};
    j["solver"] = goal->planner.solver;
    j["details"] = false;

    body = j.dump();
    res = cli.Post(api_path.c_str(), body, "application/json");

    // Check if the request was successful
    if (res && res->status == 200) {
        std::cout << "Path planning successful!" << std::endl;
        std::cout << "Response: " << res->body << std::endl;

        // Parse the JSON response
        try {
            json response = json::parse(res->body);

            // Extract data
            auto path = response["path"].get<std::vector<std::vector<int>>>();
            double cost = response["cost"];
            std::string map_id = response["map_id"];
            std::string solver_used = response["solver_used"];
            json solver_details = response["solver_details"]; // could be null
            json metrics = response["metrics"];
            double planning_time = metrics["planning_time"];

            result->quantum_metadata.total_cost = cost;
            result->planning_time.sec = static_cast<int>(planning_time);
            result->planning_time.nanosec = static_cast<int>((planning_time - static_cast<int>(planning_time)) * 1e9);
            std::string timestamp = metrics["timestamp"];

            // Print results
            std::cout << "Cost: " << cost << std::endl;
            std::cout << "Map ID: " << map_id << std::endl;
            std::cout << "Solver Used: " << solver_used << std::endl;

            std::cout << "Path:" << std::endl;
            result->path.header.stamp = this->now();
            result->path.header.frame_id = goal->goal.header.frame_id;
            for (const auto& point : path) {
                std::cout << "  [" << point[0] << ", " << point[1] << ", " << point[2] << "]" << std::endl;
                geometry_msgs::msg::PoseStamped p;
                p.pose.position.x = point[0];
                p.pose.position.y = point[1];
                p.pose.position.z = point[2];
                result->path.poses.push_back(p);
            }
        }
        catch (const std::exception& e) {
            std::cerr << "JSON parsing error: " << e.what() << std::endl;
        }
    } else {
        std::cout << "Path planning failed or no response" << std::endl;
        if (res) {
            std::cout << "Status: " << res->status << std::endl;
        } else {
            std::cout << "No response received (check connection/server)" << std::endl;
        }
        goal_handle->abort(result);
    }

    // Goal succeeded
    goal_handle->succeed(result);
    RCLCPP_INFO(this->get_logger(), "Goal succeeded");

}

std::string ServerNode::read_file(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) return "";

    std::string content;
    file.seekg(0, std::ios::end);
    content.resize(file.tellg());
    file.seekg(0, std::ios::beg);
    file.read(&content[0], content.size());
    file.close();

    return content;
}

} // namespace quantum_planner_server
