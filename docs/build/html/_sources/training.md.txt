# Training Models
Here's a detailed guide to help you navigate through model training and data processing.

The fundamental scripts used for training models are neatly placed within the `training/src` directory.

## Provided Dataset
If you wish to use the provided dataset (Processed_Data.zip, available in the dataset section), train your custom model using `train_grasp_model.py`.
```bash
cd training
python src/train_grasp_model.py
```
If you want to use wandb then you can uncomment code blocks to enable it.

## Training with Custom Simulation Data
First, pre-process the data using `process_raw_files.py`. <br/>
Then, utilize `train_grasp_model.py` for training.<br/>
```bash
cd training
python src/pre_processing_data_scripts/process_raw_files.py
python src/train_grasp_model.py
```