from setuptools import setup
import os
from glob import glob

package_name = 'ros2_arm_control'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include all launch files
        (os.path.join('share', package_name, 'launch'), glob('launch/*launch.[pxy][yma]*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Saad',
    maintainer_email='saad@example.com',
    description='Simple ROS 2 package for controlling a humanoid arm',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'arm_controller = ros2_arm_control.arm_controller:main',
            'arm_client = ros2_arm_control.arm_client:main',
        ],
    },
)