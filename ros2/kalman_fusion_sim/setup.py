from setuptools import find_packages, setup

package_name = "kalman_fusion_sim"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/sim_fusion.launch.py"]),
        ("share/" + package_name + "/worlds", ["worlds/fusion_world.sdf"]),
        ("share/" + package_name + "/config", ["config/bridge.yaml"]),
    ],
    install_requires=["setuptools", "numpy"],
    zip_safe=True,
    maintainer="sensor-fusion-lab",
    maintainer_email="dev@example.com",
    description="Gazebo (gz sim) + mock-driver simulation feeding the kalman_fusion node.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "mock_driver = kalman_fusion_sim.mock_driver_node:main",
            "odom_to_position = kalman_fusion_sim.odom_to_position_node:main",
        ],
    },
)
