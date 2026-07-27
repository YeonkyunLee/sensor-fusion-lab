"""kalman_fusion_sim — Gazebo (gz sim) + mock-driver simulation for kalman_fusion.

Drives the existing ``kalman_fusion`` node from either a real Gazebo Harmonic/
Ionic world (via ros_gz_bridge) or a pure-Python mock sensor source (headless,
CI-tested). See the package README for the two run paths.
"""
