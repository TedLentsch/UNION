# Conda environment install instructions



## UNION-Env
Create and activate an environment named UNION-Env using the commands below.
You may need to install ``gcc`` by running the command ``sudo apt-get install gcc`` to be able to build wheels for pycocotools (required for nuscenes-devkit).

```
conda env create -f environment.yml
conda activate UNION-Env
```



## openmmlab
Create and activate environment named ``openmmlab`` using commands below. After this, mmdetection3d needs to be cloned. This is described in the main README from UNION.

```
conda create --name openmmlab python=3.8
conda activate openmmlab
```

```
conda install pytorch=2.1 torchvision=0.16 torchaudio=2.1 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install fsspec=2024.6
conda install numpy=1.23
```

```
pip install -U openmim
mim install mmengine==0.9.0
mim install mmcv==2.1.0
mim install mmdet==3.2.0
```
