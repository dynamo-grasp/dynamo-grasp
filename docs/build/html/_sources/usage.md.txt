# Getting Started

## Data Collection
Before executing the `dynamo_grasp.sh` script, you must create a folder within the Dynamo-Grasp directory. This folder will serve as the storage location for data collected from the simulation. To create this folder, use the following command:
```bash
mkdir <PATH_TO_YOUR_WORKSPACE>/Dynamo-Grasp/scenario_grasp_configurations
```

Kickstart your experience with a basic usage example:

**Usage**:
```bash
./dynamo_grasp.sh --num-envs NUM_ENVS
```
**Parameters**:
```bash
NUM ENVS :  Integer representing the number of environments to run.
            Select this number based on your computational resources.
```