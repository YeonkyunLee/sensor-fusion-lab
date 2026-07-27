from setuptools import find_packages, setup

package_name = "kalman_fusion"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/fusion.launch.py"]),
    ],
    install_requires=["setuptools", "numpy"],
    zip_safe=True,
    maintainer="sensor-fusion-lab",
    maintainer_email="dev@example.com",
    description="ROS2 node wrapping a linear Kalman filter for position + IMU fusion.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "fusion_node = kalman_fusion.fusion_node:main",
        ],
    },
)
