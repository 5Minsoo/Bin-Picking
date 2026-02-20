from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import TimerAction # 이거 추가
def generate_launch_description():
    return LaunchDescription([
        Node(
            package="bin_picking",
            executable="perception_bridge",
            name="perception_bridge",
            output="screen",
        ),
        TimerAction(period=1.0, actions=[
        Node(
            package="bin_picking",
            executable="perception_planning_scene",
            name="perception_planning_scene",
            output="screen",
        )]),
        # Node(
        #     package="bin_picking",
        #     executable="object_visualization",
        #     name="object_visualization",
        #     output="screen",
        # )
        ])
