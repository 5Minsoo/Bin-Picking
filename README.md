# Model-Based Bin Picking & Grasping

A grasping pipeline for recognized objects using pre-defined grasp poses, pose classification, MoveIt collision planning, and grasp ordering. 
Supports both **real-world execution** (eye-to-hand calibration) and **Isaac Sim simulation**.

---

## Pipeline

1. **Pre-define grasp poses** per object — candidate grasps are specified relative to the object frame
2. **Classify object pose** — determine orientation class based on the object's central axis
3. **Register collision objects** — add recognized objects and environment to MoveIt planning scene
4. **Determine grasp order** — prioritize targets by Z-height, proximity to other objects, wall distance, etc.
5. **Execute grasp** — plan and execute via MoveIt with collision avoidance

---

## 1. Pre-defined Grasp Poses

Candidate grasp poses are defined relative to each object's coordinate frame. For each recognized object, the system selects a feasible grasp from the pre-defined set based on the current object pose.
<p align="center">
<img width="400" alt="Image" src="https://github.com/user-attachments/assets/d6c91302-4c67-4183-99e5-039ce1a97dd3" />
</p>

---

## 2. Pose Classification

Object orientation is classified based on its **central axis pose**. This determines which set of pre-defined grasps is applicable for the current configuration.

<p align="center">
<img width="500” alt="Image" src="https://github.com/user-attachments/assets/b4fb9968-d24f-43e7-a4e6-cc5ee0987b1f" />


</p>

---

## 3. MoveIt Collision Object Registration

All recognized objects and environmental geometry (bin walls, floor, neighboring objects) are registered as collision objects in the MoveIt planning scene. This ensures the planner generates collision-free trajectories.

<p align="center">
<img width="500” alt="Image" src="https://github.com/user-attachments/assets/31030074-87df-4560-92b8-4661c0e4acd9" />
  <br>
  <em>Registered as collision objects in RViz</em>
</p>

---

## 4. Grasp Ordering

When multiple objects are detected, the system determines the optimal grasp sequence considering:

- **Z-height** — pick higher objects first
- **Surrounding objects** — prefer isolated targets
- **Wall distance** — avoid targets too close to bin edges

<p align="center">
<img width="500" alt="Image" src="https://github.com/user-attachments/assets/b42c90cb-c161-4958-b477-b64f4cdd6532" />
  <br>
  <em>Surrounding objects</em>
</p>

---

## 5. Execution

### Real World

Tesing in real world, Using **[eye-to-hand calibration](https://github.com/5Minsoo/Handeye-Calibration)** to transform object poses from camera frame to robot base frame for actual grasping.

### Isaac Sim

Logic Tested in NVIDIA Isaac Sim with stacked/cluttered object scenarios.

<p align="center">
<img width="400" height="225" alt="Image" src="https://github.com/user-attachments/assets/242d0d85-7bab-4bc8-9898-f1e312c89e86" />
</p>

<p align="center">
<img width="400" height="294" alt="Image" src="https://github.com/user-attachments/assets/a8a84529-bfc4-4db2-a657-74a5e77a47c9" />


</p>



---

## References

- [MoveIt 2 Documentation](https://moveit.picknik.ai/)
- [NVIDIA Isaac Sim](https://developer.nvidia.com/isaac-sim)

