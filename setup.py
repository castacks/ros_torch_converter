from setuptools import find_packages, setup

package_name = 'ros_torch_converter'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test'], include=[package_name, '{}.*'.format(package_name)]),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Sam Triest',
    maintainer_email='striest@andrew.cmu.edu',
    description='Package for converting from ROS to torch',
    license='BSD-3-Clause',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'test_pub = ros_torch_converter.ros.test_publisher:main',
            'test_sub = ros_torch_converter.ros.test_subscriber:main',
            'debug = ros_torch_converter.ros.debug:main',
            'nav_plotter = ros_torch_converter.ros.test_matplotlib_nav_viz:main'
        ],
    },
)
