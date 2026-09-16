# Articulation in Motion: Prior-free Part Mobility Analysis for Articulated Objects By Dynamic-Static Disentanglement

## [Project page](https://haoai-1997.github.io/AiM/)

This repository contains the official implementation associated with the paper "Articulation in Motion: Prior-free Part Mobility Analysis for Articulated Objects By Dynamic-Static Disentanglement".

# 🚀 Setup
create an anaconda environment using
```
conda create -y -n AiM python=3.10
conda activate AiM

pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html
conda install cudatoolkit-dev=11.7 -c conda-forge

pip install -r requirements.txt

pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn/

```

# Dataset
Dataset could be downloaded at <a href="https://1sfu-my.sharepoint.com/personal/jla861_sfu_ca/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fjla861%5Fsfu%5Fca%2FDocuments%2FProject%2FPARIS%2Fdataset%2Ezip&parent=%2Fpersonal%2Fjla861%5Fsfu%5Fca%2FDocuments%2FProject%2FPARIS&ga=1" title="Onedrive">[Onedrive]</a>.
 
You can also generate yourself PartNetMobility data following data_generatation_PartNetMobility/blend.sh. (PartNet_Mobility <a href="https://huggingface.co/datasets/sapien-sim/PartNetMobility">[Here]</a>)

# Train
You can follow start.sh

```
python train_main.py --source_path /your_image_dataset --model_path /your_output_path --is_blender --eval --random_bg_color
```

You can also use the COLMAP dataset following <a href="https://github.com/graphdeco-inria/gaussian-splatting">[3DGS]</a>

## Acknowledgments

We sincerely thank the authors of [3D-GS](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/), [Deformable 3DGS](https://github.com/ingra14m/Deformable-3D-Gaussians/), [PARIS](https://github.com/3dlg-hcvc/paris), [DigitialTwinArt](https://github.com/NVlabs/DigitalTwinArt/), and  [ArtGS](https://github.com/YuLiu-LY/ArtGS), whose codes and datasets were used in our work.




## BibTex

```
@inproceedings{
ai2026articulation,
title={Articulation in Motion: Prior-free Part Mobility Analysis for Articulated Objects By Dynamic-Static Disentanglement},
author={Hao Ai and Wenjie Chang and Jianbo Jiao and Ales Leonardis and Eyal Ofek},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=mvKM40zDyn}
}
```
