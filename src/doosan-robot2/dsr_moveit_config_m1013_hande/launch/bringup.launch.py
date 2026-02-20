import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction, GroupAction, RegisterEventHandler
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import Command, FindExecutable, PathJoinSubstitution, LaunchConfiguration, PythonExpression
from launch.event_handlers import OnProcessStart, OnProcessExit
from launch_ros.actions import Node, SetRemap
from launch.actions import SetEnvironmentVariable
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import EnvironmentVariable, PathJoinSubstitution
from launch_ros.parameter_descriptions import ParameterValue
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node

from ament_index_python.packages import get_package_share_directory
from moveit_configs_utils import MoveItConfigsBuilder
# export GZ_SIM_RESOURCE_PATH=$GZ_SIM_RESOURCE_PATH:$(ros2 pkg prefix robotiq_hande_description)/share export GZ_SIM_MODEL_PATH=$GZ_SIM_MODEL_PATH:$(ros2 pkg prefix robotiq_hande_description)/share # 구버전 호환용(있어도 무해) export IGN_GAZEBO_RESOURCE_PATH=$IGN_GAZEBO_RESOURCE_PATH:$(ros2 pkg prefix robotiq_hande_description)/share
def generate_launch_description():
    # 1. Launch Arguments
    ARGUMENTS = [
        DeclareLaunchArgument('name', default_value='', description='NAME_SPACE'),
        DeclareLaunchArgument('model', default_value='m1013_hande', description='ROBOT_MODEL'),
        DeclareLaunchArgument('host', default_value='192.168.56.1', description='ROBOT_IP'),
        DeclareLaunchArgument('port', default_value='12345', description='ROBOT_PORT'),
        DeclareLaunchArgument('use_gazebo', default_value='false', description='Start Gazebo'),
        DeclareLaunchArgument('use_isaac', default_value='false', description='Start Isaac Sim'),
        DeclareLaunchArgument('use_real', default_value='true', description='Use Real Robot'),
        DeclareLaunchArgument(
            'use_sim_time',
            default_value=PythonExpression([
                '"false" if "', LaunchConfiguration('use_real'), '" == "true" else "true"'
            ]),
            description='Use sim time'
        ),
        DeclareLaunchArgument('x', default_value='0', description='Gazebo x'),
        DeclareLaunchArgument('y', default_value='0', description='Gazebo y'),
        DeclareLaunchArgument('z', default_value='0', description='Gazebo z'),
        DeclareLaunchArgument('R', default_value='0', description='Gazebo Roll'),
        DeclareLaunchArgument('P', default_value='0', description='Gazebo Pitch'),
        DeclareLaunchArgument('Y', default_value='0', description='Gazebo Yaw'),
        DeclareLaunchArgument('remap_tf', default_value='false', description='Remap TF for multi-robot'),
    ]
    # Load Configurations
    use_sim_time = LaunchConfiguration('use_sim_time')
    use_real = LaunchConfiguration('use_real')
    use_gazebo = LaunchConfiguration('use_gazebo')
    
    # 2. MoveIt Configuration
    moveit_config = (
        MoveItConfigsBuilder("m1013_hande", package_name="dsr_moveit_config_m1013_hande")
        .robot_description(file_path="config/m1013_hande.urdf.xacro")
        .robot_description_semantic(file_path="config/m1013_hande.srdf")
        .trajectory_execution(file_path="config/moveit_controllers.yaml")
        .to_moveit_configs()
    )


    # 3. Robot Description (Xacro Command)
    # real, gazebo, isaac 플래그를 모두 xacro로 전달
    robot_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]), " ",
            PathJoinSubstitution([FindPackageShare("dsr_moveit_config_m1013_hande"), "config", LaunchConfiguration('model')]), ".urdf.xacro", " ",
            "use_gazebo:=", LaunchConfiguration('use_gazebo'), " ",
            "use_isaac:=", LaunchConfiguration('use_isaac'), " ",
            "use_real:=", LaunchConfiguration('use_real'), " ",
        ]
    )
    
    robot_description = {"robot_description": ParameterValue(robot_description_content, value_type=str)}

    # 4. Common Nodes
    
    # Robot State Publisher
    node_robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="both",
        parameters=[robot_description, {"use_sim_time": use_sim_time}],
    )

    # RViz2
    rviz_config_file = os.path.join(
    get_package_share_directory('utils'),
    'rviz_config.rviz'
    )
    print("RVIZ CONFIG FILE: ", rviz_config_file)

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        arguments=["-d", rviz_config_file],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.planning_pipelines,
            moveit_config.robot_description_kinematics,
            {"use_sim_time": use_sim_time},
        ],
    )
    planning_scene_visualization = Node(
    package='utils',
    executable='planning_scene_visualization',
    name='gz_entities_to_moveit_scene',
    parameters=[
        {'update_hz': 1.0},
        {'col_cyl_segments': 12},
    ],
    output='screen',   condition=IfCondition(use_gazebo)
    )   
    # Move Group
    run_move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            moveit_config.to_dict(),
            {"use_sim_time": use_sim_time},
        ],
    )

    # 5. Real Robot Specific Nodes
    # ros2_control_node는 'use_real'일 때만 실행 (Gazebo는 플러그인이 매니저 역할)
    ros2_controllers_path = os.path.join(
        get_package_share_directory("dsr_moveit_config_m1013_hande"),
        "config",
        "ros2_controllers.yaml",
    )
    
    ros2_control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[ros2_controllers_path, robot_description,{"use_sim_time": use_sim_time}],
        output="both",
        condition=UnlessCondition(use_gazebo),
    )

    # 6. Gazebo Specific Nodes
    # Gazebo Simulator
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [FindPackageShare("ros_gz_sim"), "/launch/gz_sim.launch.py"]
        ),
        launch_arguments={"gz_args": " -r -v 3 empty.sdf"}.items(),
        condition=IfCondition(use_gazebo),
    )

    # Spawn Entity in Gazebo
    gz_spawn_entity = Node(
        package="ros_gz_sim",
        executable="create",
        output="screen",
        arguments=[
            "-topic", "robot_description",
            "-name", LaunchConfiguration('model'),
            "-allow_renaming", "true",
            "-x", LaunchConfiguration('x'), "-y", LaunchConfiguration('y'), "-z", LaunchConfiguration('z'),
            "-R", LaunchConfiguration('R'), "-P", LaunchConfiguration('P'), "-Y", LaunchConfiguration('Y'),
        ],
        condition=IfCondition(use_gazebo),
    )
    clock_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        output="screen",
        arguments=[
            "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
        ],
        condition=IfCondition(use_gazebo),
    )

    # Delete Entity Service Client in Gazebo
    # Clock Bridge (Gazebo Sim Time -> ROS Time)
    delete_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        output="screen",
        arguments=[
            "/world/empty/remove@ros_gz_interfaces/srv/DeleteEntity",
            "--ros-args",
            "-r", "/world/empty/remove:=/delete_entity",
        ],        condition=IfCondition(use_gazebo),
    )
    # 7. Controller Spawners (Common)
    # Controller names must match those in ros2_controllers.yaml
    joint_state_broadcaster = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster", "--controller-manager", "/controller_manager"],
        parameters=[{"use_sim_time": use_sim_time}],
    )

    dsr_controller = Node(
        package="controller_manager",
        executable="spawner",
        # 사용자의 YAML에 정의된 컨트롤러 이름 확인 필요 (dsr_moveit_controller or arm_controller)
        arguments=["dsr_moveit_controller", "--controller-manager", "/controller_manager"], 
        parameters=[{"use_sim_time": use_sim_time}],
    )

    gripper_controller = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["gripper_controller", "--controller-manager", "/controller_manager"],
        parameters=[{"use_sim_time": use_sim_time}],
    )

    # 8. Event Handlers & Delays
    
    # RViz는 Move Group 실행 후 3초 뒤에 실행
    delay_rviz = RegisterEventHandler(
        OnProcessStart(
            target_action=run_move_group_node,
            on_start=[TimerAction(period=3.0, actions=[rviz_node])],
        )
    )

    # Spawner 실행 타이밍 조절
    # Real: Control Node가 뜬 후 실행
    # Gazebo: Spawn이 된 후 실행 (약간의 딜레이 필요)
    delay_spawners_real = RegisterEventHandler(
        OnProcessStart(
            target_action=ros2_control_node,
            on_start=[
                TimerAction(period=2.0, actions=[
                    joint_state_broadcaster, 
                    dsr_controller, 
                    gripper_controller
                ])
            ]
        ),
        condition=UnlessCondition(use_gazebo)
    )

    # Gazebo는 Control Node가 없으므로 Timer로 넉넉히 대기 후 실행
    delay_spawners_gazebo = TimerAction(
        period=10.0, # Gazebo 켜지는 시간 고려
        actions=[
            joint_state_broadcaster, 
            dsr_controller, 
            gripper_controller,
        ],
        condition=IfCondition(use_gazebo)
    )

    gz_pose_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/world/empty/pose/info@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V',
        ],
        remappings=[
            ('/world/empty/pose/info', '/gz_pose_tf'),
        ],
        output='screen',        condition=IfCondition(use_gazebo),
    )

    
    nodes_to_start = [
        # Gazebo Environment
        gazebo,
        gz_spawn_entity,
        delete_bridge,
        gz_pose_bridge,
        planning_scene_visualization,
        clock_bridge,
        # # Real Environment
        ros2_control_node,

        # # Common
        # run_move_group_node,
        # delay_rviz,
        node_robot_state_publisher,
        
         # Move Group
        # Spawners (Conditioned delays)
        delay_spawners_real,
        delay_spawners_gazebo,
    ]

    return LaunchDescription(ARGUMENTS + nodes_to_start)