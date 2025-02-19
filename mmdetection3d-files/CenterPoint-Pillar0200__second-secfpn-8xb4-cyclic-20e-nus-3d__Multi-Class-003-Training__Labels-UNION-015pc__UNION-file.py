### Author: Ted Lentsch
### Original file: centerpoint_pillar02_second_secfpn_8xb4-cyclic-20e_nus-3d.py
### Version: mmdet3d==1.4.0



_base_ = ['../_base_/datasets/nus-3d.py',
          '../_base_/models/CenterPoint-Pillar0200__second-secfpn-nus__Multi-Class-015pc-Model__UNION-file.py',
          '../_base_/schedules/cyclic-20e.py',
          '../_base_/default_runtime.py',
         ]



### Variables.
dataset_type      = 'NuScenesDataset'
data_root         = 'data/nuscenes/'
class_names       = ['pseudoclass000', 'pseudoclass001', 'pseudoclass002', 'pseudoclass003', 'pseudoclass004', 'pseudoclass005', 'pseudoclass006', 'pseudoclass007', 'pseudoclass008', 'pseudoclass009', 'pseudoclass010', 'pseudoclass011', 'pseudoclass012', 'pseudoclass013', 'pseudoclass014']
metainfo          = dict(classes=class_names)
input_modality    = dict(use_lidar=True, use_camera=False)
backend_args      = None
point_cloud_range = [-51.2,-51.2,-5.0,51.2,51.2,3.0]



### Multi-class object detection.
data_prefix = dict(pts='samples/LIDAR_TOP', img='', sweeps='sweeps/LIDAR_TOP')
model       = dict(data_preprocessor=dict(voxel_layer=dict(point_cloud_range=point_cloud_range)),
                   pts_voxel_encoder=dict(point_cloud_range=point_cloud_range),
                   pts_bbox_head=dict(bbox_coder=dict(pc_range=point_cloud_range[:2])),
                   # Model training and testing settings.
                   train_cfg=dict(pts=dict(point_cloud_range=point_cloud_range)),
                   test_cfg=dict(pts=dict(pc_range=point_cloud_range[:2])))



### Training and testing pipelines.
train_pipeline = [dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=5, use_dim=5, backend_args=backend_args),
                  dict(type='LoadPointsFromMultiSweeps', sweeps_num=10, use_dim=[0,1,2,3,4], pad_empty_sweeps=True, remove_close=True, backend_args=backend_args),
                  dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
                  ### No dataset batch sampler.
                  dict(type='GlobalRotScaleTrans', rot_range=[-0.3925,0.3925], scale_ratio_range=[0.95,1.05], translation_std=[0,0,0]),
                  dict(type='RandomFlip3D', sync_2d=False, flip_ratio_bev_horizontal=0.5, flip_ratio_bev_vertical=0.5),
                  dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
                  dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
                  dict(type='ObjectNameFilter', classes=class_names),
                  dict(type='PointShuffle'),
                  dict(type='Pack3DDetInputs', keys=['points','gt_bboxes_3d','gt_labels_3d'])
                 ]
test_pipeline  = [dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=5, use_dim=5, backend_args=backend_args),
                  dict(type='LoadPointsFromMultiSweeps', sweeps_num=10, use_dim=[0,1,2,3,4], pad_empty_sweeps=True, remove_close=True, backend_args=backend_args),
                  dict(type='MultiScaleFlipAug3D', img_scale=(1333,800), pts_scale_ratio=1, flip=False, transforms=[dict(type='GlobalRotScaleTrans', rot_range=[0,0], scale_ratio_range=[1.,1.], translation_std=[0,0,0]), dict(type='RandomFlip3D')]),
                  dict(type='Pack3DDetInputs', keys=['points'])
                 ]



### Training, testing, and evaluation dataloaders.
train_dataloader = dict(batch_size=4, num_workers=4, persistent_workers=True,
                        sampler=dict(type='DefaultSampler', shuffle=True),
                        dataset=dict(type=dataset_type,
                                     data_root=data_root,
                                     ann_file='nuscenes_infos_train__Multi-Class-003__Labels-UNION-015pc.pkl',
                                     pipeline=train_pipeline,
                                     metainfo=metainfo,
                                     modality=input_modality,
                                     test_mode=False,
                                     data_prefix=data_prefix,
                                     box_type_3d='LiDAR',
                                     backend_args=backend_args))
val_dataloader   = dict(batch_size=1, num_workers=1, persistent_workers=True, drop_last=False,
                        sampler=dict(type='DefaultSampler', shuffle=False),
                        dataset=dict(type=dataset_type,
                                     data_root=data_root,
                                     ann_file='nuscenes_infos_val__Multi-Class-003__Labels-GT.pkl',
                                     pipeline=test_pipeline,
                                     metainfo=metainfo,
                                     modality=input_modality,
                                     test_mode=True,
                                     data_prefix=data_prefix,
                                     box_type_3d='LiDAR',
                                     backend_args=backend_args))
test_dataloader  = dict(batch_size=1, num_workers=1, persistent_workers=True, drop_last=False,
                        sampler=dict(type='DefaultSampler', shuffle=False),
                        dataset=dict(type=dataset_type,
                                     data_root=data_root,
                                     ann_file='nuscenes_infos_val__Multi-Class-003__Labels-GT.pkl',
                                     pipeline=test_pipeline,
                                     metainfo=metainfo,
                                     modality=input_modality,
                                     data_prefix=data_prefix,
                                     test_mode=True,
                                     box_type_3d='LiDAR',
                                     backend_args=backend_args))



### Evaluators.
val_evaluator  = dict(type='NuScenesMetric',
                      data_root=data_root,
                      ann_file=data_root+'nuscenes_infos_val__Multi-Class-003__Labels-GT.pkl',
                      metric='bbox',
                      backend_args=backend_args,
                      jsonfile_prefix='work_dirs/CenterPoint-Pillar0200__second-secfpn-8xb4-cyclic-20e-nus-3d__Multi-Class-003-Training__Labels-UNION-015pc__UNION-file')
test_evaluator = val_evaluator



### Training configuration.
train_cfg = dict(by_epoch=True, max_epochs=20, val_interval=20)
