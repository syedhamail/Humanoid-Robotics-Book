from setuptools import setup
import os
from glob import glob

package_name = 'vla_robot'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Saad',
    maintainer_email='saad@example.com',
    description='Vision-Language-Action Robot Control Package',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'whisper_node = vla_robot.whisper_node:main',
            'llm_planner_node = vla_robot.llm_planner_node:main',
            'multi_modal_node = vla_robot.multi_modal_node:main',
            'vla_main = vla_robot.main_integration:main',
        ],
    },
)