# UNION: Unsupervised 3D Object Detection using Appearance-based Pseudo-Classes [NeurIPS 2024]



[[`arXiv`](https://arxiv.org/abs/2405.15688)] [[`BibTeX`](#citation-information)]



+ [2024-12-11] *Poster presentation at NeurIPS 2024!*
+ [2024-10-31] *Camera-ready release on arXiv (v2)!*
+ [2024-09-25] *UNION has been accepted for NeurIPS 2024!*
+ [2024-05-24] *Paper release on arXiv (v1)!*



### Pipeline overview
![](readme-data/UNION-pipeline-overview.png)



### Conda environment

Create and activate environment named ``UNION-Env`` using commands below.
You may need to install ``gcc`` by running command ``sudo apt-get install gcc`` to be able to build wheels for pycocotools (required for nuscenes-devkit).

```
conda env create -f conda/environment.yml
conda activate UNION-Env
```



### Download nuScenes

[nuScenes](https://arxiv.org/abs/1903.11027) dataset can be downloaded [here](https://www.nuscenes.org/nuscenes).



### Generate pseudo-labels

UNION pipeline is implemented in Jupyter notebook ``UNION-pipeline__Get-mobile-objects__nuScenes.ipynb``.
Start JupyterLab with UNION-Env conda enviroment activated and execute entire notebook to discover mobile objects.

```
jupyter lab
```



### Generate MMDetection3D files
Coming soon!



### Results on nuScenes [class-agnostic detection]
Class-agnostic 3D object detection performance on [nuScenes](https://arxiv.org/abs/1903.11027) validation split (150 scenes).
For each object discovery method, [CenterPoint](https://arxiv.org/pdf/2006.11275) has been trained with method's generated pseudo-bounding boxes on [nuScenes](https://arxiv.org/abs/1903.11027) training split (700 scenes).
AAE is set to 1.0 by default for all methods.
_L_ and _C_ stand for _LiDAR_ and _camera_, respectively.
_ST_ stands for _self-training_.

| Method       | Conference                                                       | Labels | ST                            | AP ↑     | NDS ↑    | ATE ↓     | ASE ↓     | AOE ↓     | AVE ↓     |
|--------------|------------------------------------------------------------------|--------|-------------------------------|----------|----------|-----------|-----------|-----------|-----------|
| HDBSCAN      | [JOSS'17](https://joss.theoj.org/papers/10.21105/joss.00205.pdf) | L      | :negative_squared_cross_mark: | 13.8     | 15.9     | **0.574** | 0.522     | 1.601     | 1.531     |
| OYSTER       | [CVPR'23](https://arxiv.org/pdf/2311.02007)                      | L      | :ballot_box_with_check:       |  9.1     | 11.5     | 0.784     | 0.521     | 1.514     | -         |
| LISO         | [ECCV'24](https://arxiv.org/pdf/2403.07071)                      | L      | :ballot_box_with_check:       | 10.9     | 13.9     | 0.750     | **0.409** | 1.062     | -         |
| UNION (ours) | [NeurIPS'24](https://arxiv.org/pdf/2405.15688)                   | L+C    | :negative_squared_cross_mark: | **38.4** | **31.2** | 0.589     | 0.497     | **0.874** | **0.836** |



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
| HDBSCAN+SP        | L      |  5.0      | 13.0      | 14.6      |  0.4      | **0.0**   | 1.3               |
| UNION+SP          | L+C    | 12.7      | 19.7      | **34.8**  |  3.4      | **0.0**   | 1.6               |
| UNION-05pc (ours) | L+C    | **24.0**  | **24.0**  | 30.3      | **41.6**  | **0.0**   | 0.8               |
| UNION-10pc (ours) | L+C    | 19.9      | 21.7      | 27.3      | 32.5      | **0.0**   | 0.5               |
| UNION-15pc (ours) | L+C    | 18.5      | 21.2      | 25.7      | 29.9      | **0.0**   | 0.4               |
| UNION-20pc (ours) | L+C    | 17.9      | 21.7      | 23.7      | 29.9      | **0.0**   | **4.2**           |



### Citation Information
<p align="justify">
If UNION is useful to your research, please kindly recognize our contributions by citing our paper.
</p>

```
@inproceedings{lentsch2024union,
  title={{UNION}: Unsupervised {3D} Object Detection using Object Appearance-based Pseudo-Classes},
  author={Lentsch, Ted and Caesar, Holger and Gavrila, Dariu M},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2024}
}
```
