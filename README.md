<h1 align="center">
Explainable AI for Image-Based 2D/3D Registration Quality Assessment
  <img src="figures/model.png" alt="Model architecture." width="800">
  <br>
</h1>

This repository contains the algorithmic component for **Explainable AI for Image-Based 2D/3D Registration Quality Assessment**. As surgical workflows increasingly integrate advanced imaging, robotics, and algorithms, human operators remain essential for verifying system outputs to ensure patient safety. We focus on 2D/3D registration, a critical step that aligns intraoperative 2D images with preoperative 3D data for surgical navigation, where even small errors can have serious consequences. Our framework implements an explainable AI (XAI) model trained to predict registration quality and provide interpretable feedback, supporting operator decision-making. The dataset used will be released upon acceptance. 

---

## Installation

### 1. Create a new conda environment

```bash
conda create -n XAI-2D3D-RegQuality python==3.9
conda activate XAI-2D3D-RegQuality
pip install -e .
```

### 2. Install the dependencies

```bash
conda env update -f environment.yml
```

## Usage

### Python Script

To run hyperparameter optimization: 

```bash
bash optuna_hpo.sh 
```

To train model: 

```bash
bash train.sh 
```

To evaluate model:

```bash
bash inference.sh 
```

To generate Grad-CAM heatmaps:

```bash
bash gradcam.sh 
```

## Citation

If you use XAI-2D3D-RegQuality in your research, please consider citing our paper:
