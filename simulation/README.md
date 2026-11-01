# Simulation tiers

## Tier 1 (implemented, primary study): `src/dasf/simulation`, `src/dasf/sensors`

Purpose-built Python simulator, chosen because the study needs ~10^5 seed-controlled,
paired method runs with exact ground-truth reliability labels and degradation injected at a
known physical stage, on a student laptop.

* Worlds: `W1_office`, `W2_corridor`, `W3_featurepoor` (held out from training).
* Robot: differential drive, TurtleBot3 Waffle Pi-like parameters, pure-pursuit on a roadmap,
  exact unicycle integration at 100 Hz.
* Camera: level pinhole, 160x96, HFOV 62.2 deg, 3x3 supersampled procedural textures,
  auto-exposure gain, shot/read noise, 8-bit quantisation.
* LiDAR: 360 beams ray-cast against wall segments, 0.12-3.5 m, 5 Hz, Gaussian range noise.
* IMU: gyro z + 2-axis accelerometer, 100 Hz, bias random walks.
* Encoders: 4096 ticks/rev, 50 Hz, per-episode radius/base calibration error.

## Tier 2 (specified, NOT implemented): ROS 2 + Gazebo replication

Required before any claim beyond simulation. Plan:

1. ROS 2 (Humble/Jazzy) + Gazebo with the official `turtlebot3_gazebo` Waffle Pi model
   (camera, LDS-01 LiDAR, IMU, wheel odometry already modelled).
2. Degradation injection as ROS 2 nodes between the simulated driver topics and the
   estimator: `/camera/image_raw` (darkness, blur, noise, occlusion, freeze),
   `/scan` (dropout, noise, sector mask, decimation), `/imu` (bias, noise, freeze),
   `/odom` or `/joint_states` (slip scale, noise, freeze), driven by the same
   condition strings as `configs/matrix.yaml`.
3. Estimator node: port `src/dasf/fusion` (EKF) and `src/dasf/confidence` (features +
   estimator, exported weights) to a single `rclpy` node; ground truth from Gazebo
   `/model_states`.
4. Record rosbags per episode; convert to the Tier-1 `.npz` episode schema so that
   `experiments/05-08` run unchanged.
5. Report Tier-1 vs Tier-2 results side by side; a large discrepancy is itself a finding.

## Tier 3 (optional): real TurtleBot3 with motion capture ground truth.
