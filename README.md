# UNION: Unsupervised 3D Object Detection using Appearance-based Pseudo-Classes [NeurIPS'24]



[[`arXiv`](https://arxiv.org/abs/2405.15688)] [[`BibTeX`](#citation-information)]



+ [2024-10-31] *Camera-ready release on arXiv (v2)!*
+ [2024-09-25] *UNION has been accepted for NeurIPS'24!*
+ [2024-05-24] *Paper release on arXiv (v1)!*



### Installation
Coming soon!



### Generate pseudo-labels
Coming soon!



### Results on nuScenes [class-agnostic detection]
Class-agnostic 3D object detection performance on the nuScenes validation split (150 scenes).
For each object discovery method, the detector CenterPoint has been trained with the method's generated pseudo-bounding boxes on the nuScenes training split (700 scenes).
The AAE is set to 1.0 by default for all methods.

| Method           | Labels       | Self-Training                 | AP ↑     | NDS ↑    | ATE ↓     | ASE ↓     | AOE ↓     | AVE ↓     |
|------------------|--------------|-------------------------------|----------|----------|-----------|-----------|-----------|-----------|
| HDBSCAN          | LiDAR        | :negative_squared_cross_mark: | 13.8     | 15.9     | **0.574** | 0.522     | 1.601     | 1.531     |
| OYSTER           | LiDAR        | :ballot_box_with_check:       | 9.1      | 11.5     | 0.784     | 0.521     | 1.514     | -         |
| LISO             | LiDAR        | :ballot_box_with_check:       | 10.9     | 13.9     | 0.750     | **0.409** | 1.062     | -         |
| UNION (ours)     | LiDAR+Camera | :negative_squared_cross_mark: | **38.4** | **31.2** | 0.589     | 0.497     | **0.874** | **0.836** |



### Results on nuScenes [multi-class detection]
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
