# Dataset
## Raw Dataset
Running the `'dynamo_grasp.sh` script will generate raw data in the `scenario_grasp_configurations` folder. An example set, `test_grasp_data` is provided to illustrate the format of the raw data. The raw data is organized as follows:

```bash
depth_image_*.npy - Depth image of the scene
json_data_*.json - Json file containing the properties of grasp point, force and objects in the scene
rgb_*.npy - RGB image(numpy format) of the scene
rgb_*.png - RGB image(png format) of the scene
segmask_*.npy - Segmentation mask of the scene
```
depth_image_<**env_id**>\_<**config_count**>.npy <br/>
Here `env_id` is the environment id, and `config_count` is the number of configurations for that environment. <br/>
json_data_<**env_id**>\_<**grasp_count**>\_<**config_count**>.json <br/>
Here `grasp_count` indicates the number of grasp points for those configurations. <br/>

## Provided Dataset

Link to the dataset: [Prcessed_Data.zip](https://drive.google.com/file/d/16CvCuETpmtBYbqEcMIOIJCxDMaO8c4Uu/view?usp=sharing) <br/>
This dataset consists of inputs that include 4 channels: 3D point clouds and a segmentation mask. Grasp labels consist of a zero-background image with float values at each sampled point.

The dataset is organized as follows:

```bash
*_input_data_*.npy - Input data and corresponding label image is *_label_*.npy
*_input_augment1_*.npy - Augmented input data (adding noise to point cloud) and corresponding label image is *_label_*.npy
*_input_augment2_*.npy - Augmented input data (flipping the input image), corresponding flipped label image is *_label_flip_*.npy
```

File naming adheres to the following pattern: <br/>
<**file_order**>_input_data\_<**env_id**>\_<**config_count**>.npy <br/>
Here `file_order` refers to the order of the paths added in the `process_raw_files.py` file. <br/>
The same order is followed for augmented data.

`Processed_Data.zip` dataset is obtained by running the `process_raw_files.py` file after collecting raw data from the simulation.

In our simulation setup, we use two cameras: one for main sampling points and another embedded in the suction to more accurately measure suction deformation score. For dataset collection, we utilize images from the back camera.

