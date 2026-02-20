from glob import glob
import os
from setuptools import find_packages, setup

package_name = 'bin_picking'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test'],include=['bin_picking']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),(os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': ['perception_bridge = bin_picking.perception_bridge:main'
                            , 'perception_planning_scene = bin_picking.perception_planning_scene:main',
                            'grasp_planner = bin_picking.grasp_planner:main',
                            'planning_node = bin_picking.planning_node:main',
                            'object_visualization = bin_picking.object_visualization:main'
        ],
    },
)
