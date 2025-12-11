from setuptools import setup
import os
from glob import glob

package_name = 'isaac_perception'

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
    description='Isaac Perception Package for Humanoid Robotics',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'vslam_node = isaac_perception.vslam_node:main',
            'humanoid_env = isaac_perception.humanoid_env:main',
            'isaac_lab_exercise = isaac_perception.lab_exercise:main',
        ],
    },
)