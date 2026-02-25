from setuptools import find_packages, setup

package_name = 'utils'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
            ('share/' + package_name, ['rviz_config.rviz']),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='minsoo',
    maintainer_email='minsoo@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': ['gazebo_block_spawn=utils.gazebo_block_spawn:main',
                            'object_list_checking=utils.object_list_checking:main',
                            'isaac_block_spawn=utils.isaac_block_spawn:main',
                            'planning_scene_visualization=utils.planning_scene_visualization:main',
                            'joint_state_debug=utils.joint_state_debug:main',
                            'moveit_test=utils.moveit_test:main',
                            'gripper_keyboard=utils.gripper_keyboard:main',
                            'entities_to_rviz_markers=utils.entities_to_rviz_markers:main',
                            'pick_block=utils.pick_block:main',
                            "total_debug=utils.total_debug:main",
                            'planning_debug=utils.planning_debug:main',
                            'box_detect=utils.box_detect:main',
                            'real_scene=utils.planning_scene_real:main'
        ],
    },
)
