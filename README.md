# UNION: Unsupervised 3D Object Detection using Appearance-based Pseudo-Classes [NeurIPS 2024]



[[`arXiv`](https://arxiv.org/abs/2405.15688)] [[`BibTeX`](#citation-information)]



+ [2024-12-11] *Poster presentation at NeurIPS 2024!*
+ [2024-10-31] *Camera-ready release on arXiv (v2)!*
+ [2024-09-25] *UNION has been accepted for NeurIPS 2024!*
+ [2024-05-24] *Paper release on arXiv (v1)!*



### Pipeline overview
![](readme-data/UNION-pipeline-overview.png)



### Installation
Coming soon!



### Generate pseudo-labels
Coming soon!



### Generate MMDetection3D files
Coming soon!



### Results on nuScenes [class-agnostic detection]
Class-agnostic 3D object detection performance on the [nuScenes](https://arxiv.org/abs/1903.11027) validation split (150 scenes).
For each object discovery method, the detector [CenterPoint](https://arxiv.org/pdf/2006.11275) has been trained with the method's generated pseudo-bounding boxes on the [nuScenes](https://arxiv.org/abs/1903.11027) training split (700 scenes).
The AAE is set to 1.0 by default for all methods.
_L_ and _C_ stand for _LiDAR_ and _camera_, respectively.

| Method           | Conference                                                       | Labels | Self-Training                 | AP ↑     | NDS ↑    | ATE ↓     | ASE ↓     | AOE ↓     | AVE ↓     |
|------------------|------------------------------------------------------------------|--------|-------------------------------|----------|----------|-----------|-----------|-----------|-----------|
| HDBSCAN          | [JOSS'17](https://joss.theoj.org/papers/10.21105/joss.00205.pdf) | L      | :negative_squared_cross_mark: | 13.8     | 15.9     | **0.574** | 0.522     | 1.601     | 1.531     |
| OYSTER           | [CVPR'23](https://arxiv.org/pdf/2311.02007)                      | L      | :ballot_box_with_check:       |  9.1     | 11.5     | 0.784     | 0.521     | 1.514     | -         |
| LISO             | [ECCV'24](https://arxiv.org/pdf/2403.07071)                      | L      | :ballot_box_with_check:       | 10.9     | 13.9     | 0.750     | **0.409** | 1.062     | -         |
| UNION (ours)     | [NeurIPS'24](https://arxiv.org/pdf/2405.15688)                   | L+C    | :negative_squared_cross_mark: | **38.4** | **31.2** | 0.589     | 0.497     | **0.874** | **0.836** |



### Results on nuScenes [multi-class detection]
Multi-class 3D object detection performance on the [nuScenes](https://arxiv.org/abs/1903.11027) validation split (150 scenes).
For each object discovery method, the detector [CenterPoint](https://arxiv.org/pdf/2006.11275) has been trained with the method's generated pseudo-bounding boxes on the [nuScenes](https://arxiv.org/abs/1903.11027) training split (700 scenes), and class-agnostic predictions are assigned to real classes based on their size, i.e. size prior (SP).
The classes vehicle, pedestrian, and cyclist are used (see paper for more details).
The AAE is set to 1.0 by default for all methods and classes.
UNION-Xpc stands for UNION trained with X pseudo-classes.
_L_ and _C_ stand for _LiDAR_ and _camera_, respectively.

Coming soon!



### Citation Information
<p align="justify">
If UNION is useful to your research, please kindly recognize our contributions by citing our paper.
</p>

```
@article{lentsch2024union,
  title={{UNION}: Unsupervised {3D} Object Detection using Object Appearance-based Pseudo-Classes},
  author={Lentsch, Ted and Caesar, Holger and Gavrila, Dariu M},
  journal={Advances in Neural Information Processing Systems},
  year={2024}
}
```
