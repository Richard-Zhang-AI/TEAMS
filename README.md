

This repository contains the official implementation of **TEAMS: Text-prompted spatiotEmporal dual-heAd Mamba Snake**.  

## Environment Setup

Create a Conda environment and install dependencies (ensure the PyTorch CUDA version matches your system CUDA, e.g., CUDA 9.0 with torch 1.1.0):

```bash
conda create -n snake python=3.8
conda activate snake

pip install torch==1.1.0 -f https://download.pytorch.org/whl/cu90/stable
pip install Cython==0.28.2
pip install -r requirements.txt
```

Compile the CUDA extensions under `lib/csrc` (replace `ROOT` with your project root):

```bash
ROOT=/path/to/snake
cd $ROOT/lib/csrc
export CUDA_HOME="/usr/local/cuda-9.0"

cd dcn_v2 && python setup.py build_ext --inplace
cd ../extreme_utils && python setup.py build_ext --inplace
cd ../roi_align_layer && python setup.py build_ext --inplace
```

## Data and Model Preparation

All required resources (**checkpoint, `lib` folder, and dataset**) can be downloaded from Hugging Face:  
[TEAMS on Hugging Face](https://huggingface.co/Richard-ZZZZZ/TEAMS/tree/main).

- **Checkpoint**: download `ckpt_raos.pth` and place it under  
  `/data/model`
- **Dataset**: download the dataset from the same page or the original dataset link, and place it under  
  `/datasets/`
- **lib directory**: download the `lib` folder and place it under the project root  
  `/lib/`  
  After this, the path `lib/networks/clinical_bert/main_cb` used in the config should be valid.

## Configuration

Edit `configs/sbd_snake.yaml` and ensure the following fields point to the correct locations:

- **model_dir**: should point to `/data/model` or the actual directory that stores the checkpoint.
- **test.img_path**: should point to the dataset path under `/datasets`.
- **test.visual_save_root**: directory where test visualizations and logs will be saved.
- **model_clinical_bert**: should point to `lib/networks/clinical_bert/main_cb` (or your actual path).

After saving the configuration, you can run the test.

## Testing

From the project root, run:

```bash
python test.py
```

## Results

- All metrics, logs, and visualizations will be saved to the directory specified by `test.visual_save_root` in `configs/sbd_snake.yaml`.
- After testing finishes, inspect that directory to evaluate and visualize the model performance.
