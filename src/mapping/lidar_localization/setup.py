from setuptools import setup

package_name = 'lidar_localization'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Nova Team',
    maintainer_email='project.nova@utdallas.edu',
    description='Opt-in LiDAR localization for CARLA testing',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'localization_gpsguess = lidar_localization.localization_gpsguess:main',
        ],
    },
)
