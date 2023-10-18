# Robotiq 2F-85

This package contains the URDF files describing version 4 of Robotiq Adaptive Robot Gripper 2F-85. If you have a gripper made before November 2018, it's likely that you have an earlier hardware revision and should look at older versions of this package. See the [release announcement](https://dof.robotiq.com/discussion/1404/robotiq-releases-version-4-of-its-2f-85-2f-140-adaptive-grippers) for an image of this gripper and browse the [support archive](https://robotiq.com/support/archive) for old meshes.

Also included is the cable-to-wrist coupling for direct interface with e-Series URs (part number GRP-ES-CPL-062). Note that [this part has different dimensions than older couplings](https://dof.robotiq.com/discussion/1342/robotiq-material-update-robotiq-grippers-coupling).

To test the gripper URDF description type 

```
roslaunch robotiq_2f_85_gripper_visualization view_robotiq_2f_85.launch 
```