# UNION: Unsupervised 3D Object Detection using Appearance-based Pseudo-Classes [NeurIPS 2024]



[[`arXiv`](https://arxiv.org/abs/2405.15688)] [[`Hugging Face`](https://huggingface.co/datasets/TedLentsch/nuScenes-UNION-labels)] [[`BibTeX`](#citation-information)]



+ [2025-08-21] *Pseudo-labels release on Hugging Face!*
+ [2025-02-19] *Full code release finished!*
+ [2025-02-19] *Updated version on arXiv (v3)!*
+ [2024-12-11] *Poster presentation at NeurIPS 2024!*
+ [2024-10-31] *Camera-ready release on arXiv (v2)!*
+ [2024-09-25] *UNION has been accepted for NeurIPS 2024!*
+ [2024-05-24] *Paper release on arXiv (v1)!*



### Pipeline overview
![](figures/figure1-plots/figure1.jpg)

The pipeline builds on top of multiple open source projects.
In step 1, [RANSAC](https://github.com/scikit-learn/scikit-learn) is used for ground removal (BSD-3-Clause) and [HDBSCAN](https://github.com/scikit-learn-contrib/hdbscan) is used for spatial clustering (BSD-3-Clause).
Step 2 uses [ICP-Flow](https://github.com/yanconglin/ICP-Flow) to get motion estimates (Apache-2.0 license).
Lastly, step 3 uses [DINOv2](https://github.com/facebookresearch/dinov2) for encoding the camera images (Apache-2.0 license).



### Conda environment
Create and activate environment named ``UNION-Env`` using the commands below.
You may need to install ``gcc`` by running the command ``sudo apt-get install gcc`` to be able to build wheels for pycocotools (required for nuscenes-devkit).
In addition, we require mamba for installing the environment which can be installed by ``conda install mamba -n base -c conda-forge``.

```
mamba env create -f conda/environment.yml
conda activate UNION-Env
```



### DINOv2 repo
The latest DINOv2 version requires Python 3.10+. Our ``UNION-Env`` uses Python 3.8, so we pin a compatible DINOv2 commit and load it locally via ``torch.hub`` (``source='local'``).
```
cd PUT_YOUR_DIRECTORY_HERE_TO_UNION
git clone https://github.com/facebookresearch/dinov2.git
cd dinov2
git checkout 85a24602099d397264d5b30461ad7f3bfd726ca1
```



### Download nuScenes
The [nuScenes](https://arxiv.org/abs/1903.11027) dataset can be downloaded [here](https://www.nuscenes.org/nuscenes).



### Generate pseudo-labels with UNION
UNION pipeline is implemented in Jupyter notebook ``UNION-pipeline__Get-mobile-objects__nuScenes.ipynb``.
Start JupyterLab with ``UNION-Env`` conda enviroment activated and execute entire notebook to discover mobile objects.

```
conda activate UNION-Env
jupyter lab
```



### Generate MMDetection3D files for training
Create and activate environment named ``openmmlab`` using commands below.

```
mamba create --name openmmlab python=3.8
conda activate openmmlab
```

```
mamba install pytorch=2.1 torchvision=0.16 torchaudio=2.1 pytorch-cuda=12.1 -c pytorch -c nvidia
mamba install fsspec=2024.6
mamba install numpy=1.23
```

```
pip install -U openmim
mim install mmengine==0.9.0
mim install mmcv==2.1.0
mim install mmdet==3.2.0
```

Clone the [mmdetection3d](https://github.com/open-mmlab/mmdetection3d) repository and checkout version v1.4 using the commands below.
After that, install the package.

```
cd PUT_YOUR_DIRECTORY_HERE_TO_UNION
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout fe25f7a51d36e3702f961e198894580d83c4387b
pip install -v -e .
```

Make a soft link for nuScenes in the data folder of [mmdetection3d](https://github.com/open-mmlab/mmdetection3d).
After that, process the dataset to get the ``nuscenes_infos_train.pkl`` and ``nuscenes_infos_val.pkl`` files.

```
cd PUT_YOUR_DIRECTORY_HERE_TO_UNION
cd mmdetection3d
ln -s PUT_YOUR_DIRECTORY_HERE_TO_NUSCENES/nuscenes data/nuscenes
python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes
```

When the UNION pipeline has been executed, the [mmdetection3d](https://github.com/open-mmlab/mmdetection3d) files for UNION can be generated.
This is implemented in Jupyter notebook ``UNION-pipeline__Generate-mmdet3d-files__nuScenes.ipynb``.
Start JupyterLab with ``UNION-Env`` conda enviroment activated and execute entire notebook.

```
conda activate UNION-Env
jupyter lab
```

The MMDetection3D files can also be found on [Hugging Face](https://huggingface.co/datasets/TedLentsch/nuScenes-UNION-labels).



### Train with MMDetection3D and evaluate
Some files need to be added to the [mmdetection3d](https://github.com/open-mmlab/mmdetection3d) repository and some need to be replaced.
This can be done using commands below.
The files are located in the [mmdetection3d-files](mmdetection3d-files) folder.

```
cp mmdetection3d-files/CenterPoint*Training*.py mmdetection3d/configs/centerpoint/
cp mmdetection3d-files/CenterPoint*Model*.py mmdetection3d/configs/_base_/models/
cp mmdetection3d-files/nuscenes_metric.py mmdetection3d/mmdet3d/evaluation/metrics/nuscenes_metric.py
cp mmdetection3d-files/nuscenes_dataset.py mmdetection3d/mmdet3d/datasets/nuscenes_dataset.py
```

Train [CenterPoint](https://arxiv.org/pdf/2006.11275) using the created .pkl files.

```
conda activate openmmlab
cd mmdetection3d
```

All training commands follow below (i.e. one command per experiment):

```
python tools/train.py configs/centerpoint/CenterPoint-Pillar0200__second-secfpn-8xb4-cyclic-20e-nus-3d__Class-Agnostic-Training__Labels-GT__UNION-file.py
python tools/train.py configs/centerpoint/CenterPoint-Pillar0200__second-secfpn-8xb4-cyclic-20e-nus-3d__Class-Agnostic-Training__Labels-HDBSCAN__UNION-file.py
python tools/train.py configs/centerpoint/CenterPoint-Pillar0200__second-secfpn-8xb4-cyclic-20e-nus-3d__Class-Agnostic-Training__Labels-Scene-Flow__UNION-file.py
python tools/train.py configs/centerpoint/CenterPoint-Pillar0200__second-secfpn-8xb4-cyclic-20e-nus-3d__Class-Agnostic-Training__Labels-UNION__UNION-file.py
python tools/train.py configs/centerpoint/CenterPoint-Pillar0200__second-secfpn-8xb4-cyclic-20e-nus-3d__Multi-Class-003-Training__Labels-GT__UNION-file.py
python tools/train.py configs/centerpoint/CenterPoint-Pillar0200__second-secfpn-8xb4-cyclic-20e-nus-3d__Multi-Class-005pc-Training__Labels-UNION-005pc__UNION-file.py
python tools/train.py configs/centerpoint/CenterPoint-Pillar0200__second-secfpn-8xb4-cyclic-20e-nus-3d__Multi-Class-010pc-Training__Labels-UNION-010pc__UNION-file.py
python tools/train.py configs/centerpoint/CenterPoint-Pillar0200__second-secfpn-8xb4-cyclic-20e-nus-3d__Multi-Class-015pc-Training__Labels-UNION-015pc__UNION-file.py
python tools/train.py configs/centerpoint/CenterPoint-Pillar0200__second-secfpn-8xb4-cyclic-20e-nus-3d__Multi-Class-020pc-Training__Labels-UNION-020pc__UNION-file.py
```

When the trainings have been done, the results can be computed.
This is implemented in Jupyter notebook ``UNION-pipeline__Do-evaluation-after-training__nuScenes.ipynb``.
Start JupyterLab with ``UNION-Env`` conda enviroment activated and execute entire notebook.

```
conda activate UNION-Env
jupyter lab
```



### Results on nuScenes [class-agnostic detection]
Class-agnostic 3D object detection performance on [nuScenes](https://arxiv.org/abs/1903.11027) validation split (150 scenes).
For each object discovery method, [CenterPoint](https://arxiv.org/pdf/2006.11275) has been trained with method's generated pseudo-bounding boxes on [nuScenes](https://arxiv.org/abs/1903.11027) training split (700 scenes).
AAE is set to 1.0 by default for all methods.
_L_ and _C_ stand for _LiDAR_ and _camera_, respectively.
_ST_ stands for _self-training_.

| Method       | Conference                                                       | Labels | ST                            | AP ↑     | NDS ↑    | ATE ↓     | ASE ↓     | AOE ↓     | AVE ↓     |
|--------------|------------------------------------------------------------------|--------|-------------------------------|----------|----------|-----------|-----------|-----------|-----------|
| HDBSCAN      | [JOSS'17](https://joss.theoj.org/papers/10.21105/joss.00205.pdf) | L      | :negative_squared_cross_mark: | 13.8     | 15.7     | **0.583** | 0.531     | 1.517     | 1.556     |
| OYSTER       | [CVPR'23](https://arxiv.org/pdf/2311.02007)                      | L      | :ballot_box_with_check:       |  9.1     | 11.5     | 0.784     | 0.521     | 1.514     | -         |
| LISO         | [ECCV'24](https://arxiv.org/pdf/2403.07071)                      | L      | :ballot_box_with_check:       | 10.9     | 13.9     | 0.750     | **0.409** | 1.062     | -         |
| UNION (ours) | [NeurIPS'24](https://arxiv.org/pdf/2405.15688)                   | L+C    | :negative_squared_cross_mark: | **39.5** | **31.7** | 0.590     | 0.506     | **0.876** | **0.837** |



### Results on nuScenes [multi-class detection]
Multi-class 3D object detection performance on [nuScenes](https://arxiv.org/abs/1903.11027) validation split (150 scenes).
For each object discovery method, [CenterPoint](https://arxiv.org/pdf/2006.11275) has been trained with the method's generated pseudo-bounding boxes on [nuScenes](https://arxiv.org/abs/1903.11027) training split (700 scenes), and class-agnostic predictions are assigned to real classes based on their size, i.e. size prior (SP).
Vehicle (Veh.), pedestrian (Ped.), and cyclist (Cyc.) classes are used, see paper for more details.
AAE is set to 1.0 by default for all methods and classes.
_UNION-Xpc_ stands for UNION trained with X pseudo-classes.
_L_ and _C_ stand for _LiDAR_ and _camera_, respectively.
&dagger;Without clipping precision-recall curve, clipping is default for [nuScenes](https://arxiv.org/abs/1903.11027) evaluation.

| Method            | Labels | mAP ↑     | NDS ↑     | Veh. AP ↑ | Ped. AP ↑ | Cyc. AP ↑ | Cyc. AP&dagger; ↑ |
|-------------------|--------|-----------|-----------|-----------|-----------|-----------|-------------------|
| HDBSCAN+SP        | L      |  4.9      | 12.8      | 14.1      |  0.4      | **0.0**   | 1.5               |
| UNION+SP          | L+C    | 13.0      | 19.7      | **35.2**  |  3.7      | **0.0**   | 1.5               |
| UNION-05pc (ours) | L+C    | **25.1**  | **24.4**  | 31.0      | **44.2**  | **0.0**   | 0.7               |
| UNION-10pc (ours) | L+C    | 20.4      | 22.1      | 27.6      | 33.7      | **0.0**   | 0.5               |
| UNION-15pc (ours) | L+C    | 18.9      | 21.2      | 25.6      | 31.1      | **0.0**   | 0.4               |
| UNION-20pc (ours) | L+C    | 19.0      | 21.9      | 25.1      | 31.9      | **0.0**   | **2.2**           |



### Citation Information
<p align="justify">
If UNION is useful to your research, please kindly recognize our contributions by citing our paper.
</p>

```
@inproceedings{lentsch2024union,
  title={{UNION}: Unsupervised {3D} Object Detection using Object Appearance-based Pseudo-Classes},
  author={Lentsch, Ted and Caesar, Holger and Gavrila, Dariu M},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  pages={22028--22046},
  volume={37},
  year={2024}
}
```
