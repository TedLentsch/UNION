import cv2
import k3d
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torchvision
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from nuscenes.utils.splits import train, val, test, mini_train, mini_val
from pyquaternion import Quaternion
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from typing import Dict, List
from utils.utils_functions import get_lidar_sweep



def plot_axes(T_plotorigin_target:torch.Tensor=torch.eye(4),):
    """
    Creates k3d axes representation with red, green, blue vectors representing x, y, z axis of target frame within plot's origin frame.
    
    Args:
        T_plotorigin_target (torch.Tensor) : Homogeneous transformation matrix for mapping 3D points from target frame to plotorigin frame.
        
    Return:
        pose_axes (k3d.vectors) : Object representing axes of target frame.
    """
    
    
    length       = 1.0
    unit_vectors = torch.tensor([[length,0,0,1], [0,length,0,1], [0,0,length,1]])
    start        = torch.stack((T_plotorigin_target[:,3], T_plotorigin_target[:,3], T_plotorigin_target[:,3],), dim=0)
    end          = (T_plotorigin_target @ unit_vectors.T).T
    pose_axes    = k3d.vectors(origins=start[:,:3], vectors=(end-start)[:,:3], colors=[0xFF0000, 0xFF0000, 0x00FF00, 0x00FF00, 0x0000FF, 0x0000FF],)    
    
    
    return pose_axes



def plot_manual_annotations(nusc:NuScenes, scene_idx:int, sample_idx:int, scenes:List, mobile_classes:Dict, annot_range_thres:float, num_lidar_points_thres:int,):
    """
    Visualize manual annotations.
    
    Args:
        nusc (NuScenes) : NuScenes Object.
        scene_idx (int) : Scene index.
        sweep_idx (int) : Sweep index.
        scenes (list) : List with scene dictionaries of dataset.
        mobile_classes (dict) : Dictionary containing mobile classes.
        annot_range_thres (float) : Range threshold for annotations to be shown.
        num_lidar_points_thres (int) : Minimum number of LiDAR points for annotations to be shown.
    """
    
    
    lidar_token = scenes[scene_idx]['sample_lidar_tokens'][sample_idx]
    
    
    # Get LiDAR points.
    sweep_pc_lidar = get_lidar_sweep(nusc=nusc, lidar_token=lidar_token)
    
    
    # Get transforms and calculate pc_global.
    T_global_vehicle = scenes[scene_idx]['sample_lidar_T_global_vehicle'][sample_idx]
    T_vehicle_lidar  = scenes[scene_idx]['T_vehicle_lidar']
    pc_global        = (T_global_vehicle @ T_vehicle_lidar @ sweep_pc_lidar.T).T
    
    
    # Ego-vehicle pose and LiDAR sensor pose.
    lidar_record                = nusc.get('sample_data', lidar_token)
    sample_lidar_egopose_record = nusc.get('ego_pose', lidar_record['ego_pose_token'])
    lidar_sensorpose_record     = nusc.get('calibrated_sensor', lidar_record['calibrated_sensor_token'])
    
    
    # Process annotations.
    box_list = []
    for annot_token in scenes[scene_idx]['sample_annotations'][sample_idx]:
        annot_record = nusc.get('sample_annotation', annot_token)
        # Get box in global frame of scene (box_global).
        box = Box(annot_record['translation'], annot_record['size'], Quaternion(annot_record['rotation']), name=annot_record['category_name'], token=annot_record['token'])
        # Transform box from global frame to vehicle frame (T_vehicle_global).
        box.translate(-np.array(sample_lidar_egopose_record['translation']))
        box.rotate(Quaternion(sample_lidar_egopose_record['rotation']).inverse)
        # Transform box from vehicle frame to LiDAR frame (T_lidar_vehicle).
        box.translate(-np.array(lidar_sensorpose_record['translation']))
        box.rotate(Quaternion(lidar_sensorpose_record['rotation']).inverse)
        # Filter for (1) class, (2) distance to LiDAR, and (3) number of LiDAR points.
        if annot_record['category_name'] in list(mobile_classes.keys()):
            class_range_thres = 40 if mobile_classes[box.name] in ['pedestrian','motorcycle','bicycle'] else 50
            if np.linalg.norm(box.center.copy()[:2], ord=2)<=min(annot_range_thres, class_range_thres):
                if annot_record['num_lidar_pts']>=num_lidar_points_thres:
                    box_list.append(box)
                    
                    
    # Get bounding box colors.
    box_colors_rgb = torch.stack([torch.tensor(nusc.colormap[box.name]) for box in box_list])
    box_colors_k3d = (box_colors_rgb @ torch.tensor([[2**16,2**8,2**0]]).T).flatten().tolist()
    
    
    # Plot.
    BOX_MESH_INDICES = torch.IntTensor([[0, 1, 2], [0, 2, 3], [0, 1, 5], [0, 5, 4],
                                        [2, 3, 7], [2, 7, 6], [1, 2, 6], [1, 6, 5],
                                        [0, 3, 7], [0, 7, 4], [4, 5, 6], [4, 6, 7],])
    
    pc_lidar = sweep_pc_lidar.clone()
    
    plot = k3d.plot(camera_auto_fit=False, grid_visible=False, menu_visibility=False, axes_helper=0.0)
    plot += plot_axes()
    
    point_range_thres = 55   # Unit: meters.
    bool_range        = torch.linalg.norm(pc_lidar[:,:2], dim=1)<=point_range_thres
    plot += k3d.points(positions=pc_lidar[bool_range,:3].float(), point_size=0.20, color=0xa3a4ad)
    
    for annot_idx in range(len(box_list)):
        corners = torch.from_numpy(box_list[annot_idx].corners().T).float()
        plot += k3d.mesh(vertices=corners, indices=BOX_MESH_INDICES, color=box_colors_k3d[annot_idx], opacity=0.8)
        
    plot.camera = [-36.394487547828334,-23.785169471359588,45.685273053840966,10.951503363386365,-4.996362939441929,-9.35579383068484,0.6324026859848865,0.2898148170869679,0.7183830555879935]
    plot.display()
    
    print(f'Sample has {len(box_list)} annotations after filtering!')
    print('This frame is part of Train split!') if scenes[scene_idx]['scene_name'] in train else print('This frame is part of Val split!')



def plot_ground_segmented_sweep(nusc:NuScenes, scene_idx:int, sweep_idx:int, scenes:List, intermediate_results_groundremoval_dir1:str,):
    """
    Visualize ground removal results.
    
    Args:
        nusc (NuScenes) : NuScenes Object.
        scenes (list) : List with scene dictionaries of dataset.
        scene_idx (int) : Scene index.
        sweep_idx (int) : Sweep index.
        intermediate_results_groundremoval_dir1 (str) : Folder for ground removal results (boolean arrays).
    """
    
    
    lidar_token = scenes[scene_idx]['sweep_lidar_tokens'][sweep_idx]
    
    
    # Get LiDAR points.
    pc_lidar = get_lidar_sweep(nusc=nusc, lidar_token=lidar_token)
    
    
    # Get ground segmentation.
    lidar_record          = nusc.get('sample_data', lidar_token)
    segmentation_filename = os.path.basename(lidar_record['filename']).replace('.pcd.bin','.npy')
    boolall_ground        = torch.from_numpy(np.load(os.path.join(intermediate_results_groundremoval_dir1, segmentation_filename)))>0
    
    
    # Plot.
    plot = k3d.plot(camera_auto_fit=False, grid_visible=False, menu_visibility=False, axes_helper=0.0)
    plot += plot_axes()
    
    point_range_thres = 50   # Unit: meters.
    bool_range        = torch.linalg.norm(pc_lidar[:,:2], dim=1)<=point_range_thres
    plot += k3d.points(positions=pc_lidar[bool_range & boolall_ground,:3][::2].float(), point_size=0.25, color=0xa3a4ad)
    plot += k3d.points(positions=pc_lidar[bool_range & (~boolall_ground),:3][::2].float(), point_size=0.25, color=0xf5425a)
    
    plot.camera = [-36.394487547828334,-23.785169471359588,45.685273053840966,10.951503363386365,-4.996362939441929,-9.35579383068484,0.6324026859848865,0.2898148170869679,0.7183830555879935]
    plot.display()
    
    print('This frame is part of Train split!') if scenes[scene_idx]['scene_name'] in train else print('This frame is part of Val split!')



def plot_spatial_clusters(nusc:NuScenes, scenes:List, M:int, intermediate_results_spatialclustering_dir:str, scene_idx:int, sample_idx:int, dense:bool, bbox:bool,):
    """
    Visualize spatial clustering results, i.e. fitted spatial clusters and fitted bounding boxes.
    
    Args:
        nusc (NuScenes) : NuScenes Object.
        scenes (list) : List with scene dictionaries of dataset.
        M (int) : Add more frames for clustering, this results in 2*M+1 frames, i.e. [-M,M].
        indermediate_results_spatialclustering_dir (str) : Folder for spatial clustering results (cluster dicts).
        scene_idx (int) : Scene index.
        sample_idx (int) : Sample index.
        dense (bool) : Show extra sweeps.
        bbox (bool) : Show bounding boxes.
    """
    
    
    # Get sweep index.
    timestamp = scenes[scene_idx]['sample_timestamps'][sample_idx]
    sweep_idx = scenes[scene_idx]['sweep_lidar_timestamps'].index(timestamp)
    
    
    # Get sweep information.
    lidar_token          = scenes[scene_idx]['sweep_lidar_tokens'][sweep_idx]
    T_mainvehicle_global = torch.linalg.inv(scenes[scene_idx]['sweep_lidar_T_global_vehicle'][sweep_idx])
    T_vehicle_lidar      = scenes[scene_idx]['T_vehicle_lidar']
    
    
    # Get LiDAR points.
    pc_lidar = get_lidar_sweep(nusc=nusc, lidar_token=lidar_token)
    
    
    # Add more frames [-M,+M]; this results in 2*M+1 frames.
    if dense:
        sweep_ids = list(range(len(scenes[scene_idx]['sweep_lidar_tokens'])))

        past_extra_ids = list(range(max(sweep_ids.index(sweep_idx)-M,0), sweep_ids.index(sweep_idx)))
        next_extra_ids = list(range(min(sweep_ids.index(sweep_idx)+1,len(sweep_ids)), min(sweep_ids.index(sweep_idx)+1+M,len(sweep_ids))))
        if len(past_extra_ids)<M:
            next_extra_ids.extend(list(range(next_extra_ids[-1]+1, next_extra_ids[-1]+1+M-len(past_extra_ids))))
        elif len(next_extra_ids)<M:
            past_extra_ids.extend(list(range(past_extra_ids[0]-(M-len(next_extra_ids)), past_extra_ids[0])))
        extra_ids = torch.sort(torch.IntTensor(past_extra_ids+next_extra_ids)).values.tolist()
        extra_ids = torch.IntTensor(sweep_ids)[extra_ids].tolist()

        extra_pc_mainlidar_list = []
        for idx in extra_ids:
            extra_pc_lidar       = get_lidar_sweep(nusc=nusc, lidar_token=scenes[scene_idx]['sweep_lidar_tokens'][idx])
            extra_T_global_lidar = scenes[scene_idx]['sweep_lidar_T_global_vehicle'][idx] @ T_vehicle_lidar
            extra_pc_mainlidar   = (torch.linalg.inv(T_vehicle_lidar) @ T_mainvehicle_global @ extra_T_global_lidar @ extra_pc_lidar.T).T
            extra_pc_mainlidar_list.append(extra_pc_mainlidar.clone())
        if len(extra_ids)>0:
            pc_lidar = torch.concatenate((pc_lidar, torch.concatenate(extra_pc_mainlidar_list)), dim=0)
            
            
    # Get cluster dict.
    lidar_record = nusc.get('sample_data', lidar_token)
    filename     = os.path.basename(lidar_record['filename']).replace('.pcd.bin','.npy')
    cluster_dict = np.load(os.path.join(intermediate_results_spatialclustering_dir, filename), allow_pickle=True).item()
    
    
    # Plot.
    np.random.seed(0)
    
    colors_qualitative     = torch.tensor([[166, 206, 227], [31, 120, 180], [178, 223, 138], [51, 160, 44], [251, 154, 153], [227, 26, 28], [253, 191, 111], [255, 127, 0], [202, 178, 214], [106, 61, 154], [255, 255, 153], [177, 89, 40],])
    colors_qualitative_k3d = (colors_qualitative @ torch.tensor([[2**16], [2**8], [2**0]])).flatten().tolist()
    
    BOX_MESH_INDICES = torch.IntTensor([[0, 1, 2], [0, 2, 3], [0, 1, 5], [0, 5, 4],
                                        [2, 3, 7], [2, 7, 6], [1, 2, 6], [1, 6, 5],
                                        [0, 3, 7], [0, 7, 4], [4, 5, 6], [4, 6, 7],])
    
    plot = k3d.plot(camera_auto_fit=False, grid_visible=False, menu_visibility=False, axes_helper=0.0)
    plot += plot_axes()
    
    point_range_thres = 50   # Unit: meters.
    bool_range        = torch.linalg.norm(pc_lidar[:,:2], dim=1)<=point_range_thres
    
    points_plotted = torch.zeros([pc_lidar.shape[0]], dtype=torch.bool)
    for cluster_idx in cluster_dict.keys():
        if cluster_dict[cluster_idx].touchground_bboxdimensions[2]>0:
            bboxlength, bboxwidth, bboxheight = cluster_dict[cluster_idx].touchground_bboxdimensions
            T_lidar_bbox                      = torch.from_numpy(cluster_dict[cluster_idx].touchground_T_lidar_bbox)
        else:
            bboxlength, bboxwidth, bboxheight = cluster_dict[cluster_idx].bboxdimensions
            T_lidar_bbox                      = torch.from_numpy(cluster_dict[cluster_idx].T_lidar_bbox)
            
        if dense:
            plot += k3d.points(positions=pc_lidar[cluster_dict[cluster_idx].idsall_aggregated2,:3][::M].float(), point_size=0.25, color=colors_qualitative_k3d[int(len(colors_qualitative_k3d)*np.random.rand())])
            points_plotted[cluster_dict[cluster_idx].idsall_aggregated2] = True
        else:
            if len(cluster_dict[cluster_idx].idsall_frame[0])>0:
                plot += k3d.points(positions=pc_lidar[cluster_dict[cluster_idx].idsall_frame2[0],:3].float(), point_size=0.25, color=colors_qualitative_k3d[int(len(colors_qualitative_k3d)*np.random.rand())])
                points_plotted[cluster_dict[cluster_idx].idsall_frame2[0]] = True
                
        if bbox:
            bboxcorners_bbox  = torch.tensor([[ bboxlength/2,  bboxwidth/2, 0,          1],
                                              [-bboxlength/2,  bboxwidth/2, 0,          1],
                                              [-bboxlength/2, -bboxwidth/2, 0,          1],
                                              [ bboxlength/2, -bboxwidth/2, 0,          1],
                                              [ bboxlength/2,  bboxwidth/2, bboxheight, 1],
                                              [-bboxlength/2,  bboxwidth/2, bboxheight, 1],
                                              [-bboxlength/2, -bboxwidth/2, bboxheight, 1],
                                              [ bboxlength/2, -bboxwidth/2, bboxheight, 1]])
            bboxcorners_lidar = (T_lidar_bbox @ bboxcorners_bbox.T).T
            plot += k3d.mesh(vertices=bboxcorners_lidar[:,:3].float(), indices=BOX_MESH_INDICES, opacity=0.2)
            
    plot += k3d.points(positions=pc_lidar[bool_range & (~points_plotted),:3].float(), point_size=0.25, color=0xa3a4ad)
    
    plot.camera = [-36.394487547828334,-23.785169471359588,45.685273053840966,10.951503363386365,-4.996362939441929,-9.35579383068484,0.6324026859848865,0.2898148170869679,0.7183830555879935]
    plot.display()
    
    print('This frame is part of Train split!') if scenes[scene_idx]['scene_name'] in train else print('This frame is part of Val split!')



def plot_motion_status(nusc:NuScenes, scenes:List, M:int, intermediate_results_sceneflow_dir:str, scene_idx:int, sample_idx:int, dense:bool, bbox:bool, label:bool,):
    """
    Visualize the motion status, i.e. dynamic object proposals in red.
    
    Args:
        nusc (NuScenes) : NuScenes Object.
        scenes (list) : List with scene dictionaries of dataset.
        M (int) : Add more frames for clustering, this results in 2*M+1 frames, i.e. [-M,M].
        intermediate_results_sceneflow_dir (str) : Folder for scene flow results (cluster dicts).
        scene_idx (int) : Scene index.
        sample_idx (int) : Sample index.
        dense (bool) : Show extra sweeps.
        bbox (bool) : Show bounding boxes.
        label (bool) : Show cluster ID labels.
    """
    
    
    # Get sweep index.
    timestamp = scenes[scene_idx]['sample_timestamps'][sample_idx]
    sweep_idx = scenes[scene_idx]['sweep_lidar_timestamps'].index(timestamp)
    
    
    # Get sweep information.
    lidar_token          = scenes[scene_idx]['sweep_lidar_tokens'][sweep_idx]
    T_mainvehicle_global = torch.linalg.inv(scenes[scene_idx]['sweep_lidar_T_global_vehicle'][sweep_idx])
    T_vehicle_lidar      = scenes[scene_idx]['T_vehicle_lidar']
    
    
    # Get LiDAR points.
    pc_lidar = get_lidar_sweep(nusc=nusc, lidar_token=lidar_token)
    
    
    # Add more frames [-M,+M]; this results in 2*M+1 frames.
    if dense:
        sweep_ids = list(range(len(scenes[scene_idx]['sweep_lidar_tokens'])))

        past_extra_ids = list(range(max(sweep_ids.index(sweep_idx)-M,0), sweep_ids.index(sweep_idx)))
        next_extra_ids = list(range(min(sweep_ids.index(sweep_idx)+1,len(sweep_ids)), min(sweep_ids.index(sweep_idx)+1+M,len(sweep_ids))))
        if len(past_extra_ids)<M:
            next_extra_ids.extend(list(range(next_extra_ids[-1]+1, next_extra_ids[-1]+1+M-len(past_extra_ids))))
        elif len(next_extra_ids)<M:
            past_extra_ids.extend(list(range(past_extra_ids[0]-(M-len(next_extra_ids)), past_extra_ids[0])))
        extra_ids = torch.sort(torch.IntTensor(past_extra_ids+next_extra_ids)).values.tolist()
        extra_ids = torch.IntTensor(sweep_ids)[extra_ids].tolist()

        extra_pc_mainlidar_list = []
        for idx in extra_ids:
            extra_pc_lidar       = get_lidar_sweep(nusc=nusc, lidar_token=scenes[scene_idx]['sweep_lidar_tokens'][idx])
            extra_T_global_lidar = scenes[scene_idx]['sweep_lidar_T_global_vehicle'][idx] @ T_vehicle_lidar
            extra_pc_mainlidar   = (torch.linalg.inv(T_vehicle_lidar) @ T_mainvehicle_global @ extra_T_global_lidar @ extra_pc_lidar.T).T
            extra_pc_mainlidar_list.append(extra_pc_mainlidar.clone())
        if len(extra_ids)>0:
            pc_lidar = torch.concatenate((pc_lidar, torch.concatenate(extra_pc_mainlidar_list)), dim=0)
            
            
    # Get cluster dict.
    lidar_record = nusc.get('sample_data', lidar_token)
    filename     = os.path.basename(lidar_record['filename']).replace('.pcd.bin','.npy')
    cluster_dict = np.load(os.path.join(intermediate_results_sceneflow_dir, filename), allow_pickle=True).item()
    
    
    # Plot.
    np.random.seed(0)
    
    colors_qualitative     = torch.tensor([[166, 206, 227], [31, 120, 180], [178, 223, 138], [51, 160, 44], [251, 154, 153], [227, 26, 28], [253, 191, 111], [255, 127, 0], [202, 178, 214], [106, 61, 154], [255, 255, 153], [177, 89, 40],])
    colors_qualitative_k3d = (colors_qualitative @ torch.tensor([[2**16], [2**8], [2**0]])).flatten().tolist()
    
    BOX_MESH_INDICES = torch.IntTensor([[0, 1, 2], [0, 2, 3], [0, 1, 5], [0, 5, 4],
                                        [2, 3, 7], [2, 7, 6], [1, 2, 6], [1, 6, 5],
                                        [0, 3, 7], [0, 7, 4], [4, 5, 6], [4, 6, 7],])
    
    plot = k3d.plot(camera_auto_fit=False, grid_visible=False, menu_visibility=False, axes_helper=0.0)
    plot += plot_axes()
    
    point_range_thres = 50   # Unit: meters.
    bool_range        = torch.linalg.norm(pc_lidar[:,:2], dim=1)<=point_range_thres
    
    for cluster_idx in cluster_dict.keys():
        if cluster_dict[cluster_idx].sceneflow__touchground_bboxdimensions[2]>0:
            bboxlength, bboxwidth, bboxheight = cluster_dict[cluster_idx].sceneflow__touchground_bboxdimensions
            T_lidar_bbox                      = torch.from_numpy(cluster_dict[cluster_idx].sceneflow__touchground_T_lidar_bbox)
        elif cluster_dict[cluster_idx].touchground_bboxdimensions[2]>0:
            bboxlength, bboxwidth, bboxheight = cluster_dict[cluster_idx].touchground_bboxdimensions
            T_lidar_bbox                      = torch.from_numpy(cluster_dict[cluster_idx].touchground_T_lidar_bbox)
        else:
            bboxlength, bboxwidth, bboxheight = cluster_dict[cluster_idx].bboxdimensions
            T_lidar_bbox                      = torch.from_numpy(cluster_dict[cluster_idx].T_lidar_bbox)
            
        if label:
            plot += k3d.text(f'ID: {cluster_idx}', size=0.4, position=tuple(T_lidar_bbox[:3,3]))
            
        color = 0xf5425a if cluster_dict[cluster_idx].velocity_magnitude>=0.5 else 0xa3a4ad
        
        if dense:
            plot += k3d.points(positions=pc_lidar[cluster_dict[cluster_idx].idsall_aggregated2,:3][::M].float(), point_size=0.25, color=color)
        else:
            if len(cluster_dict[cluster_idx].idsall_frame[0])>0:
                plot += k3d.points(positions=pc_lidar[cluster_dict[cluster_idx].idsall_frame2[0],:3].float(), point_size=0.25, color=color)
                
        if bbox:
            bboxcorners_bbox  = torch.tensor([[ bboxlength/2,  bboxwidth/2, 0,          1],
                                              [-bboxlength/2,  bboxwidth/2, 0,          1],
                                              [-bboxlength/2, -bboxwidth/2, 0,          1],
                                              [ bboxlength/2, -bboxwidth/2, 0,          1],
                                              [ bboxlength/2,  bboxwidth/2, bboxheight, 1],
                                              [-bboxlength/2,  bboxwidth/2, bboxheight, 1],
                                              [-bboxlength/2, -bboxwidth/2, bboxheight, 1],
                                              [ bboxlength/2, -bboxwidth/2, bboxheight, 1]])
            bboxcorners_lidar = (T_lidar_bbox @ bboxcorners_bbox.T).T
            plot += k3d.mesh(vertices=bboxcorners_lidar[:,:3].float(), indices=BOX_MESH_INDICES, opacity=0.2)
            
    plot.camera = [-36.394487547828334,-23.785169471359588,45.685273053840966,10.951503363386365,-4.996362939441929,-9.35579383068484,0.6324026859848865,0.2898148170869679,0.7183830555879935]
    plot.display()
    
    print('This frame is part of Train split!') if scenes[scene_idx]['scene_name'] in train else print('This frame is part of Val split!')



def plot_appearance_similarities(nusc:NuScenes, scenes:List, M:int, intermediate_results_spatialclustering_dir:str, intermediate_results_appearanceembedding_dir:str, scene_idx:int, sample_idx:int, dense:bool, bbox:bool, cluster_idx:int, label:bool,):
    """
    Visualize appearance similarity of clusters. Reference cluster in blue and object appearance similarity indicated with red.
    
    Args:
        nusc (NuScenes) : NuScenes Object.
        scenes (list) : List with scene dictionaries of dataset.
        M (int) : Add more frames for clustering, this results in 2*M+1 frames, i.e. [-M,M].
        indermediate_results_spatialclustering_dir (str) : Folder for spatial clustering results (cluster dicts).
        intermediate_results_appearanceembedding_dir (str) : Folder for appearance embedding results (feature dicts).
        scene_idx (int) : Scene index.
        sample_idx (int) : Sample index.
        dense (bool) : Show extra sweeps.
        bbox (bool) : Show bounding boxes.
        cluster_idx (int) : Cluster index.
        label (bool) : Show cluster ID labels.
    """
    
    
    # Get sweep index.
    timestamp = scenes[scene_idx]['sample_timestamps'][sample_idx]
    sweep_idx = scenes[scene_idx]['sweep_lidar_timestamps'].index(timestamp)
    
    
    # Get sweep information.
    lidar_token          = scenes[scene_idx]['sweep_lidar_tokens'][sweep_idx]
    T_mainvehicle_global = torch.linalg.inv(scenes[scene_idx]['sweep_lidar_T_global_vehicle'][sweep_idx])
    T_vehicle_lidar      = scenes[scene_idx]['T_vehicle_lidar']
    
    
    # Get LiDAR points.
    pc_lidar = get_lidar_sweep(nusc=nusc, lidar_token=lidar_token)
    
    
    # Add more frames [-M,+M]; this results in 2*M+1 frames.
    if dense:
        sweep_ids = list(range(len(scenes[scene_idx]['sweep_lidar_tokens'])))

        past_extra_ids = list(range(max(sweep_ids.index(sweep_idx)-M,0), sweep_ids.index(sweep_idx)))
        next_extra_ids = list(range(min(sweep_ids.index(sweep_idx)+1,len(sweep_ids)), min(sweep_ids.index(sweep_idx)+1+M,len(sweep_ids))))
        if len(past_extra_ids)<M:
            next_extra_ids.extend(list(range(next_extra_ids[-1]+1, next_extra_ids[-1]+1+M-len(past_extra_ids))))
        elif len(next_extra_ids)<M:
            past_extra_ids.extend(list(range(past_extra_ids[0]-(M-len(next_extra_ids)), past_extra_ids[0])))
        extra_ids = torch.sort(torch.IntTensor(past_extra_ids+next_extra_ids)).values.tolist()
        extra_ids = torch.IntTensor(sweep_ids)[extra_ids].tolist()

        extra_pc_mainlidar_list = []
        for idx in extra_ids:
            extra_pc_lidar       = get_lidar_sweep(nusc=nusc, lidar_token=scenes[scene_idx]['sweep_lidar_tokens'][idx])
            extra_T_global_lidar = scenes[scene_idx]['sweep_lidar_T_global_vehicle'][idx] @ T_vehicle_lidar
            extra_pc_mainlidar   = (torch.linalg.inv(T_vehicle_lidar) @ T_mainvehicle_global @ extra_T_global_lidar @ extra_pc_lidar.T).T
            extra_pc_mainlidar_list.append(extra_pc_mainlidar.clone())
        if len(extra_ids)>0:
            pc_lidar = torch.concatenate((pc_lidar, torch.concatenate(extra_pc_mainlidar_list)), dim=0)
            
            
    # Get cluster dict.
    lidar_record = nusc.get('sample_data', lidar_token)
    filename     = os.path.basename(lidar_record['filename']).replace('.pcd.bin','.npy')
    cluster_dict = np.load(os.path.join(intermediate_results_spatialclustering_dir, filename), allow_pickle=True).item()
    
    
    # Get feature dict.
    lidar_record = nusc.get('sample_data', lidar_token)
    filename     = os.path.basename(lidar_record['filename']).replace('.pcd.bin','.npy')
    feature_dict = np.load(os.path.join(intermediate_results_appearanceembedding_dir, filename), allow_pickle=True).item()
    
    
    # Compute cosine similarities.
    cluster_idx = min(len(cluster_dict.keys())-1, cluster_idx)
    
    reference_feature = torch.from_numpy(feature_dict[cluster_idx].feature).reshape(1,-1)
    feature_map       = torch.from_numpy(np.array([feature_info.feature for feature_info in feature_dict.values()]))
    
    reference_feature /= torch.linalg.norm(reference_feature, ord=2)
    feature_map       /= torch.linalg.norm(feature_map, ord=2, axis=1).reshape(-1,1)
    
    cosine_sim          = (reference_feature*feature_map).sum(dim=1)
    cosine_sim__nonzero = torch.clamp(cosine_sim, min=0, max=1)
    
    
    # Plot.
    np.random.seed(0)
    
    colors_qualitative = torch.round(255*torch.vstack((cosine_sim__nonzero, torch.zeros([2,cosine_sim__nonzero.shape[0]]))).T).int()
    colors_qualitative[torch.isnan(cosine_sim__nonzero),:] = torch.tensor([135,135,135]).int()
    colors_qualitative[cluster_idx,:] = torch.tensor([0,0,255]).int()
    colors_qualitative_k3d = (colors_qualitative @ torch.tensor([[2**16], [2**8], [2**0]]).int()).flatten().tolist()
    
    BOX_MESH_INDICES = torch.IntTensor([[0, 1, 2], [0, 2, 3], [0, 1, 5], [0, 5, 4],
                                        [2, 3, 7], [2, 7, 6], [1, 2, 6], [1, 6, 5],
                                        [0, 3, 7], [0, 7, 4], [4, 5, 6], [4, 6, 7],])
    
    plot = k3d.plot(camera_auto_fit=False, grid_visible=False, menu_visibility=False, axes_helper=0.0)
    plot += plot_axes()
    
    point_range_thres = 50   # Unit: meters.
    bool_range        = torch.linalg.norm(pc_lidar[:,:2], dim=1)<=point_range_thres
    
    for cluster_idx in cluster_dict.keys():
        if cluster_dict[cluster_idx].touchground_bboxdimensions[2]>0:
            bboxlength, bboxwidth, bboxheight = cluster_dict[cluster_idx].touchground_bboxdimensions
            T_lidar_bbox                      = torch.from_numpy(cluster_dict[cluster_idx].touchground_T_lidar_bbox)
        else:
            bboxlength, bboxwidth, bboxheight = cluster_dict[cluster_idx].bboxdimensions
            T_lidar_bbox                      = torch.from_numpy(cluster_dict[cluster_idx].T_lidar_bbox)
            
        if label:
            plot += k3d.text(f'ID: {cluster_idx}', size=0.4, position=tuple(T_lidar_bbox[:3,3]))
            
        if dense:
            plot += k3d.points(positions=pc_lidar[cluster_dict[cluster_idx].idsall_aggregated2,:3][::M].float(), point_size=0.25, color=colors_qualitative_k3d[cluster_idx])
        else:
            if len(cluster_dict[cluster_idx].idsall_frame[0])>0:
                plot += k3d.points(positions=pc_lidar[cluster_dict[cluster_idx].idsall_frame2[0],:3].float(), point_size=0.25, color=colors_qualitative_k3d[cluster_idx])
                
        if bbox:
            bboxcorners_bbox  = torch.tensor([[ bboxlength/2,  bboxwidth/2, 0,          1],
                                              [-bboxlength/2,  bboxwidth/2, 0,          1],
                                              [-bboxlength/2, -bboxwidth/2, 0,          1],
                                              [ bboxlength/2, -bboxwidth/2, 0,          1],
                                              [ bboxlength/2,  bboxwidth/2, bboxheight, 1],
                                              [-bboxlength/2,  bboxwidth/2, bboxheight, 1],
                                              [-bboxlength/2, -bboxwidth/2, bboxheight, 1],
                                              [ bboxlength/2, -bboxwidth/2, bboxheight, 1]])
            bboxcorners_lidar = (T_lidar_bbox @ bboxcorners_bbox.T).T
            plot += k3d.mesh(vertices=bboxcorners_lidar[:,:3].float(), indices=BOX_MESH_INDICES, opacity=0.2)
    plot.camera = [-36.394487547828334,-23.785169471359588,45.685273053840966,10.951503363386365,-4.996362939441929,-9.35579383068484,0.6324026859848865,0.2898148170869679,0.7183830555879935]
    plot.display()
    
    print('This frame is part of Train split!') if scenes[scene_idx]['scene_name'] in train else print('This frame is part of Val split!')



def plot_velocity_fractions(hyperparameters:Dict, intermediate_results_appearanceclustering_dir:str):
    """
    Visualize velocity fractions per appearance cluster. Velocity fractions are sorted based on value. Non-mobile and mobile clusters are indicated in blue and orange, respectively.
    
    Args:
        hyperparameters (dict) : Hyperparameters for appearance clustering.
        intermediate_results_appearanceclustering_dir (str) : Folder for appearance clustering results (cluster dicts).
    """
    
    
    # Get velocity fractions.
    K__class_agnostic     = hyperparameters['Step1__K__class_agnostic']   # Unit: 1.
    moving_fraction_thres = hyperparameters['Step2__moving_fraction_thres']   # Unit: 1.
    
    filename = os.path.join(intermediate_results_appearanceclustering_dir, f'velocityfraction-per-cluster__K-class-agnostic{str(K__class_agnostic).zfill(3)}_moving-fraction-thres0{str(int(10000*moving_fraction_thres)).zfill(4)}__class_agnostic.npy')
    
    velocityfraction_per_cluster = np.load(filename, allow_pickle=True).item()
    velocityfraction_per_cluster = torch.tensor(list(velocityfraction_per_cluster.values()))
    velocityfraction_per_cluster = torch.sort(velocityfraction_per_cluster).values
    
    
    # Plot.
    blue_length = (velocityfraction_per_cluster<moving_fraction_thres).sum().item()
    
    plt.figure(figsize=[5,4])
    plt.bar(np.linspace(1,K__class_agnostic,K__class_agnostic)[:blue_length], 100*velocityfraction_per_cluster[:blue_length], color='cornflowerblue', zorder=-10)
    plt.bar(np.linspace(1,K__class_agnostic,K__class_agnostic)[blue_length:], 100*velocityfraction_per_cluster[blue_length:], color='darkorange', zorder=-10)
    plt.hlines(y=100*moving_fraction_thres, xmin=0, xmax=K__class_agnostic+1, color='red', linewidth=2, zorder=-5)
    plt.xticks(torch.linspace(0,K__class_agnostic,int(K__class_agnostic/5)+1).int().tolist(), fontsize=12)
    plt.yticks(torch.linspace(0,5*int(min(100*(velocityfraction_per_cluster[-1].item()+0.05)/5,100)),int(5*int(min(100*(velocityfraction_per_cluster[-1].item()+0.05)/5,100))/5)+1).int().tolist(), fontsize=12)
    plt.xlabel('Appearance Cluster ID', fontsize=12)
    plt.ylabel('Dynamic Object Proposal Fraction (%)', fontsize=12)
    plt.xlim(0.5,K__class_agnostic+0.5)
    plt.ylim(0,5*int(min(100*(velocityfraction_per_cluster[-1].item()+0.10)/5,100)))
    plt.text(x=3, y=100*moving_fraction_thres+0.8, s=f'Threshold = {str(int(100*moving_fraction_thres))} %', fontsize=10)
    plt.savefig(os.path.join('figure4-plots', 'velocityfractions.pdf'), bbox_inches='tight')



def plot_mobile_objects(nusc:NuScenes, scenes:List, M:int, hyperparameters:Dict, intermediate_results_sceneflow_dir:str, intermediate_results_appearanceclustering_dir:str, scene_idx:int, sample_idx:int, dense:bool, bbox:bool, label:bool,):
    """
    Visualize mobile objects.
    
    Args:
        nusc (NuScenes) : NuScenes Object.
        scenes (list) : List with scene dictionaries of dataset.
        M (int) : Add more frames for clustering, this results in 2*M+1 frames, i.e. [-M,M].
        hyperparameters (dict) : Hyperparameters for appearance clustering.
        intermediate_results_sceneflow_dir (str) : Folder for scene flow results (cluster dicts).
        intermediate_results_appearanceclustering_dir (str) : Folder for appearance clustering results (cluster dicts).
        scene_idx (int) : Scene index.
        sample_idx (int) : Sample index.
        dense (bool) : Show extra sweeps.
        bbox (bool) : Show bounding boxes.
        label (bool) : Show cluster ID labels.
    """
    
    
    if scenes[scene_idx]['scene_name'] in train:
        
        
        # Get sweep index.
        timestamp = scenes[scene_idx]['sample_timestamps'][sample_idx]
        sweep_idx = scenes[scene_idx]['sweep_lidar_timestamps'].index(timestamp)
        
        
        # Get sweep information.
        lidar_token          = scenes[scene_idx]['sweep_lidar_tokens'][sweep_idx]
        T_mainvehicle_global = torch.linalg.inv(scenes[scene_idx]['sweep_lidar_T_global_vehicle'][sweep_idx])
        T_vehicle_lidar      = scenes[scene_idx]['T_vehicle_lidar']
        
        
        # Get LiDAR points.
        pc_lidar = get_lidar_sweep(nusc=nusc, lidar_token=lidar_token)
        
        
        # Add more frames [-M,+M]; this results in 2*M+1 frames.
        if dense:
            sweep_ids = list(range(len(scenes[scene_idx]['sweep_lidar_tokens'])))
            
            past_extra_ids = list(range(max(sweep_ids.index(sweep_idx)-M,0), sweep_ids.index(sweep_idx)))
            next_extra_ids = list(range(min(sweep_ids.index(sweep_idx)+1,len(sweep_ids)), min(sweep_ids.index(sweep_idx)+1+M,len(sweep_ids))))
            if len(past_extra_ids)<M:
                next_extra_ids.extend(list(range(next_extra_ids[-1]+1, next_extra_ids[-1]+1+M-len(past_extra_ids))))
            elif len(next_extra_ids)<M:
                past_extra_ids.extend(list(range(past_extra_ids[0]-(M-len(next_extra_ids)), past_extra_ids[0])))
            extra_ids = torch.sort(torch.IntTensor(past_extra_ids+next_extra_ids)).values.tolist()
            extra_ids = torch.IntTensor(sweep_ids)[extra_ids].tolist()
            
            extra_pc_mainlidar_list = []
            for idx in extra_ids:
                extra_pc_lidar       = get_lidar_sweep(nusc=nusc, lidar_token=scenes[scene_idx]['sweep_lidar_tokens'][idx])
                extra_T_global_lidar = scenes[scene_idx]['sweep_lidar_T_global_vehicle'][idx] @ T_vehicle_lidar
                extra_pc_mainlidar   = (torch.linalg.inv(T_vehicle_lidar) @ T_mainvehicle_global @ extra_T_global_lidar @ extra_pc_lidar.T).T
                extra_pc_mainlidar_list.append(extra_pc_mainlidar.clone())
            if len(extra_ids)>0:
                pc_lidar = torch.concatenate((pc_lidar, torch.concatenate(extra_pc_mainlidar_list)), dim=0)
                
                
        # Get cluster dict.
        lidar_record = nusc.get('sample_data', lidar_token)
        filename     = os.path.basename(lidar_record['filename']).replace('.pcd.bin','.npy')
        cluster_dict = np.load(os.path.join(intermediate_results_sceneflow_dir, filename), allow_pickle=True).item()
        
        
        # Get usage dict.
        K__class_agnostic     = hyperparameters['Step1__K__class_agnostic']   # Unit: 1.
        moving_fraction_thres = hyperparameters['Step2__moving_fraction_thres']   # Unit: 1.
        
        lidar_record = nusc.get('sample_data', lidar_token)
        filename     = os.path.basename(lidar_record['filename']).replace('.pcd.bin',f'__K-class-agnostic{str(K__class_agnostic).zfill(3)}_moving-fraction-thres0{str(int(10000*moving_fraction_thres)).zfill(4)}__usage-dict__class_agnostic.npy')
        usage_dict   = np.load(os.path.join(intermediate_results_appearanceclustering_dir, filename), allow_pickle=True).item()
        
        
        # Plot.
        np.random.seed(0)
        
        colors_qualitative     = torch.tensor([[166, 206, 227], [31, 120, 180], [178, 223, 138], [51, 160, 44], [251, 154, 153], [227, 26, 28], [253, 191, 111], [255, 127, 0], [202, 178, 214], [106, 61, 154], [255, 255, 153], [177, 89, 40],])
        colors_qualitative_k3d = (colors_qualitative @ torch.tensor([[2**16], [2**8], [2**0]])).flatten().tolist()
        
        BOX_MESH_INDICES = torch.IntTensor([[0, 1, 2], [0, 2, 3], [0, 1, 5], [0, 5, 4],
                                            [2, 3, 7], [2, 7, 6], [1, 2, 6], [1, 6, 5],
                                            [0, 3, 7], [0, 7, 4], [4, 5, 6], [4, 6, 7],])
        
        plot = k3d.plot(camera_auto_fit=False, grid_visible=False, menu_visibility=False, axes_helper=0.0)
        plot += plot_axes()
        
        point_range_thres = 50   # Unit: meters.
        bool_range        = torch.linalg.norm(pc_lidar[:,:2], dim=1)<=point_range_thres
        
        points_plotted = torch.zeros([pc_lidar.shape[0]], dtype=torch.bool)
        for cluster_idx in cluster_dict.keys():
            if usage_dict[cluster_idx]:
                if cluster_dict[cluster_idx].sceneflow__touchground_bboxdimensions[2]>0:
                    bboxlength, bboxwidth, bboxheight = cluster_dict[cluster_idx].sceneflow__touchground_bboxdimensions
                    T_lidar_bbox                      = torch.from_numpy(cluster_dict[cluster_idx].sceneflow__touchground_T_lidar_bbox)
                elif cluster_dict[cluster_idx].touchground_bboxdimensions[2]>0:
                    bboxlength, bboxwidth, bboxheight = cluster_dict[cluster_idx].touchground_bboxdimensions
                    T_lidar_bbox                      = torch.from_numpy(cluster_dict[cluster_idx].touchground_T_lidar_bbox)
                else:
                    bboxlength, bboxwidth, bboxheight = cluster_dict[cluster_idx].bboxdimensions
                    T_lidar_bbox                      = torch.from_numpy(cluster_dict[cluster_idx].T_lidar_bbox)
                    
                if label:
                    plot += k3d.text(f'ID: {cluster_idx}', size=0.4, position=tuple(T_lidar_bbox[:3,3]))
                    
                if dense:
                    plot += k3d.points(positions=pc_lidar[cluster_dict[cluster_idx].idsall_aggregated2,:3][::M].float(), point_size=0.25, color=0xf5425a)
                    points_plotted[cluster_dict[cluster_idx].idsall_aggregated2] = True
                else:
                    if len(cluster_dict[cluster_idx].idsall_frame[0])>0:
                        plot += k3d.points(positions=pc_lidar[cluster_dict[cluster_idx].idsall_frame2[0],:3].float(), point_size=0.25, color=0xf5425a)
                        points_plotted[cluster_dict[cluster_idx].idsall_frame2[0]] = True
                        
                if bbox:
                    bboxcorners_bbox  = torch.tensor([[ bboxlength/2,  bboxwidth/2, 0,          1],
                                                      [-bboxlength/2,  bboxwidth/2, 0,          1],
                                                      [-bboxlength/2, -bboxwidth/2, 0,          1],
                                                      [ bboxlength/2, -bboxwidth/2, 0,          1],
                                                      [ bboxlength/2,  bboxwidth/2, bboxheight, 1],
                                                      [-bboxlength/2,  bboxwidth/2, bboxheight, 1],
                                                      [-bboxlength/2, -bboxwidth/2, bboxheight, 1],
                                                      [ bboxlength/2, -bboxwidth/2, bboxheight, 1]])
                    bboxcorners_lidar = (T_lidar_bbox @ bboxcorners_bbox.T).T
                    plot += k3d.mesh(vertices=bboxcorners_lidar[:,:3].float(), indices=BOX_MESH_INDICES, opacity=0.2)
                    
        plot += k3d.points(positions=pc_lidar[bool_range & (~points_plotted),:3].float(), point_size=0.25, color=0xa3a4ad)
        
        plot.camera = [-36.394487547828334,-23.785169471359588,45.685273053840966,10.951503363386365,-4.996362939441929,-9.35579383068484,0.6324026859848865,0.2898148170869679,0.7183830555879935]
        plot.display()
        
        print('This frame is part of Train split!')
        
        
    else:
        print('This frame is part of Val split! Val is not used for the appearance clustering! Independent data!')



def plot_qualitative_example(nusc:NuScenes, scenes:List, hyperparameters:Dict, intermediate_results_spatialclustering_dir:str, intermediate_results_sceneflow_dir:str, intermediate_results_appearanceclustering_dir:str, USE_MINI_SPLIT:bool, mobile_classes:Dict, load_reference:bool):
    """
    Visualize (a) HDBSCAN, (b) Scene Flow, (c) UNION, and (d) Ground Truth bounding boxes for scene-1100. This is Figure 3 in the paper.
    
    Args:
        nusc (NuScenes) : NuScenes Object.
        scenes (list) : List with scene dictionaries of dataset.
        hyperparameters (dict) : Hyperparameters for appearance clustering.
        indermediate_results_spatialclustering_dir (str) : Folder for spatial clustering results (cluster dicts).
        intermediate_results_sceneflow_dir (str) : Folder for scene flow results (cluster dicts).
        intermediate_results_appearanceclustering_dir (str) : Folder for appearance clustering results (cluster dicts).
        USE_MINI_SPLIT (bool) : Use mini split from nuScenes data, else entire dataset.
        mobile_classes (dict) : Mapping for the mobile detection classes of nuScenes.
        load_reference (bool) : Load reference files instead of the new computed files.
    """
    
    
    # Settings.
    scene_idx              = 9 if USE_MINI_SPLIT else 840
    sample_idx             = 2
    annot_range_thres      = 50   # Unit: meters.
    num_lidar_points_thres =  1   # Unit: 1.
    
    
    # Get LiDAR points.
    lidar_token = scenes[scene_idx]['sample_lidar_tokens'][sample_idx]
    pc_lidar    = get_lidar_sweep(nusc=nusc, lidar_token=lidar_token)
    
    
    # Get transforms and calculate pc_global.
    T_global_vehicle = scenes[scene_idx]['sample_lidar_T_global_vehicle'][sample_idx]
    T_vehicle_lidar  = scenes[scene_idx]['T_vehicle_lidar']
    
    
    # Ego-vehicle pose and LiDAR sensor pose.
    lidar_record                = nusc.get('sample_data', lidar_token)
    sample_lidar_egopose_record = nusc.get('ego_pose', lidar_record['ego_pose_token'])
    lidar_sensorpose_record     = nusc.get('calibrated_sensor', lidar_record['calibrated_sensor_token'])
    
    
    # Get pseudo-labels from UNION pipeline.
    if load_reference:
        cluster_dict__spatialclustering  = np.load(os.path.join('figure3-plots', 'spatialclustering__n015-2018-11-21-19-38-26+0800__LIDAR_TOP__1542800988948006-reference.npy'), allow_pickle=True).item()
        cluster_dict__sceneflow          = np.load(os.path.join('figure3-plots', 'sceneflow__n015-2018-11-21-19-38-26+0800__LIDAR_TOP__1542800988948006-reference.npy'), allow_pickle=True).item()
        usage_dict__appearanceclustering = np.load(os.path.join('figure3-plots', 'appearanceclustering__n015-2018-11-21-19-38-26+0800__LIDAR_TOP__1542800988948006-reference.npy'), allow_pickle=True).item()
        
    else:
        K__class_agnostic     = hyperparameters['Step1__K__class_agnostic']   # Unit: 1.
        moving_fraction_thres = hyperparameters['Step2__moving_fraction_thres']   # Unit: 1.
        
        filename__spatialclustering    = os.path.basename(lidar_record['filename']).replace('.pcd.bin','.npy')
        filename__sceneflow            = os.path.basename(lidar_record['filename']).replace('.pcd.bin','.npy')
        filename__appearanceclustering = os.path.basename(lidar_record['filename']).replace('.pcd.bin',f'__K-class-agnostic{str(K__class_agnostic).zfill(3)}_moving-fraction-thres0{str(int(10000*moving_fraction_thres)).zfill(4)}__usage-dict__class_agnostic.npy')
        
        cluster_dict__spatialclustering  = np.load(os.path.join(intermediate_results_spatialclustering_dir, filename__spatialclustering), allow_pickle=True).item()
        cluster_dict__sceneflow          = np.load(os.path.join(intermediate_results_sceneflow_dir, filename__sceneflow), allow_pickle=True).item()
        usage_dict__appearanceclustering = np.load(os.path.join(intermediate_results_appearanceclustering_dir, filename__appearanceclustering), allow_pickle=True).item()
        
        
    # Get ground-truth annotations.
    box_list, velocity_list = [], []
    for annot_token in scenes[scene_idx]['sample_annotations'][sample_idx]:
        annot_record = nusc.get('sample_annotation', annot_token)
        # Get box in global frame of scene (box_global).
        box = Box(annot_record['translation'], annot_record['size'], Quaternion(annot_record['rotation']), name=annot_record['category_name'], token=annot_record['token'])
        # Transform box from global frame to vehicle frame (T_vehicle_global).
        box.translate(-np.array(sample_lidar_egopose_record['translation']))
        box.rotate(Quaternion(sample_lidar_egopose_record['rotation']).inverse)
        # Transform box from vehicle frame to LiDAR frame (T_lidar_vehicle).
        box.translate(-np.array(lidar_sensorpose_record['translation']))
        box.rotate(Quaternion(lidar_sensorpose_record['rotation']).inverse)
        # Filter for (1) class, (2) distance to LiDAR, and (3) number of LiDAR points.
        if annot_record['category_name'] in list(mobile_classes.keys()):
            class_range_thres = 40 if mobile_classes[box.name] in ['pedestrian','motorcycle','bicycle'] else 50
            if np.linalg.norm(box.center.copy()[:2], ord=2)<=min(annot_range_thres, class_range_thres):
                if annot_record['num_lidar_pts']>=num_lidar_points_thres:
                    box_list.append(box)
                    velocity_list.append(nusc.box_velocity(annot_record['token']))
                    
                    
    # Plot 1: HDBSCAN.
    plt.figure(dpi=200, figsize=[8,8])
    plt.scatter(pc_lidar[:,0], pc_lidar[:,1], s=0.5, color='silver')
    for cluster_idx in cluster_dict__spatialclustering.keys():
        cluster = cluster_dict__spatialclustering[cluster_idx]
        T_lidar_bbox = cluster.touchground_T_lidar_bbox
        xc, yc       = T_lidar_bbox[:2,3]
        L, W         = cluster.touchground_bboxdimensions[:2]
        yaw          = cluster.touchground_yaw_radians
        corners_bbox  = np.array([[L/2, -W/2, 0, 1], [L/2, W/2, 0, 1], [-L/2, W/2, 0, 1], [-L/2, -W/2, 0, 1]])
        corners_lidar = (T_lidar_bbox @ corners_bbox.T).T
        arrow_bbox  = np.array([[0, 0, 0, 1], [L/2, 0, 0, 1]])
        arrow_lidar = (T_lidar_bbox @ arrow_bbox.T).T
        plt.plot(corners_lidar[[0,1,2,3,0],0], corners_lidar[[0,1,2,3,0],1], color='k', linewidth=2)
        plt.plot(arrow_lidar[:,0], arrow_lidar[:,1], color='k', linewidth=2)
    plt.xlim(-15,25)
    plt.ylim(-18,22)
    plt.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False, labeltop=False, labelright=False, labelbottom=False)
    plt.savefig(os.path.join('figure3-plots', '3a-hdbscan.png'), bbox_inches='tight')
    
    
    # Plot 2: Scene Flow.
    plt.figure(dpi=200, figsize=[8,8])
    plt.scatter(pc_lidar[:,0], pc_lidar[:,1], s=0.5, color='silver')
    for cluster_idx in cluster_dict__sceneflow.keys():
        cluster = cluster_dict__sceneflow[cluster_idx]
        T_lidar_bbox = cluster.sceneflow__touchground_T_lidar_bbox
        xc, yc       = T_lidar_bbox[:2,3]
        L, W         = cluster.sceneflow__touchground_bboxdimensions[:2]
        yaw          = cluster.sceneflow__touchground_yaw_radians
        v            = cluster.velocity_magnitude
        corners_bbox  = np.array([[L/2, -W/2, 0, 1], [L/2, W/2, 0, 1], [-L/2, W/2, 0, 1], [-L/2, -W/2, 0, 1]])
        corners_lidar = (T_lidar_bbox @ corners_bbox.T).T
        arrow_bbox  = np.array([[0, 0, 0, 1], [L/2, 0, 0, 1]])
        arrow_lidar = (T_lidar_bbox @ arrow_bbox.T).T
        if v<0.5:
            plt.plot(corners_lidar[[0,1,2,3,0],0], corners_lidar[[0,1,2,3,0],1], color='k', linewidth=2)
            plt.plot(arrow_lidar[:,0], arrow_lidar[:,1], color='k', linewidth=2)
        elif v>=0.5:
            plt.plot(corners_lidar[[0,1,2,3,0],0], corners_lidar[[0,1,2,3,0],1], color='red', linewidth=2)
            plt.plot(arrow_lidar[:,0], arrow_lidar[:,1], color='red', linewidth=2)
    plt.xlim(-15,25)
    plt.ylim(-18,22)
    plt.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False, labeltop=False, labelright=False, labelbottom=False)
    plt.savefig(os.path.join('figure3-plots', '3b-sceneflow.png'), bbox_inches='tight')
    
    
    # Plot3: UNION.
    plt.figure(dpi=200, figsize=[8,8])
    plt.scatter(pc_lidar[:,0], pc_lidar[:,1], s=0.5, color='silver')
    for cluster_idx in cluster_dict__sceneflow.keys():
        if usage_dict__appearanceclustering[cluster_idx]:
            cluster = cluster_dict__sceneflow[cluster_idx]
            T_lidar_bbox = cluster.sceneflow__touchground_T_lidar_bbox
            xc, yc       = T_lidar_bbox[:2,3]
            L, W         = cluster.sceneflow__touchground_bboxdimensions[:2]
            yaw          = cluster.sceneflow__touchground_yaw_radians
            v            = cluster.velocity_magnitude
            corners_bbox  = np.array([[L/2, -W/2, 0, 1], [L/2, W/2, 0, 1], [-L/2, W/2, 0, 1], [-L/2, -W/2, 0, 1]])
            corners_lidar = (T_lidar_bbox @ corners_bbox.T).T
            arrow_bbox  = np.array([[0, 0, 0, 1], [L/2, 0, 0, 1]])
            arrow_lidar = (T_lidar_bbox @ arrow_bbox.T).T
            if v<0.5:
                plt.plot(corners_lidar[[0,1,2,3,0],0], corners_lidar[[0,1,2,3,0],1], color='green', linewidth=2)
                plt.plot(arrow_lidar[:,0], arrow_lidar[:,1], color='green', linewidth=2)
            elif v>=0.5:
                plt.plot(corners_lidar[[0,1,2,3,0],0], corners_lidar[[0,1,2,3,0],1], color='red', linewidth=2)
                plt.plot(arrow_lidar[:,0], arrow_lidar[:,1], color='red', linewidth=2)
    plt.xlim(-15,25)
    plt.ylim(-18,22)
    plt.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False, labeltop=False, labelright=False, labelbottom=False)
    plt.savefig(os.path.join('figure3-plots', '3c-appearance.png'), bbox_inches='tight')
    
    
    # Plot 4: Ground Truth.
    plt.figure(dpi=200, figsize=[8,8])
    plt.scatter(pc_lidar[:,0], pc_lidar[:,1], s=0.5, color='silver')
    for box_lidar, velocity_lidar in zip(box_list, velocity_list):
        yaw    = box_lidar.orientation.yaw_pitch_roll[0]
        xc, yc = box_lidar.center[:2]
        W, L   = box_lidar.wlh[:2]
        v      = np.linalg.norm(velocity_lidar[:2], ord=2)
        c, s = np.cos(yaw), np.sin(yaw)
        pseudo_T_lidar_bbox = np.array([[c, -s, 0, xc], [s, c, 0, yc], [0, 0, 1, 0], [0, 0, 0, 1],])
        corners_bbox  = np.array([[L/2, -W/2, 0, 1], [L/2, W/2, 0, 1], [-L/2, W/2, 0, 1], [-L/2, -W/2, 0, 1]])
        corners_lidar = (pseudo_T_lidar_bbox @ corners_bbox.T).T
        arrow_bbox  = np.array([[0, 0, 0, 1], [L/2, 0, 0, 1]])
        arrow_lidar = (pseudo_T_lidar_bbox @ arrow_bbox.T).T
        plt.plot(corners_lidar[[0,1,2,3,0],0], corners_lidar[[0,1,2,3,0],1], color='blue', linewidth=2)
        plt.plot(arrow_lidar[:,0], arrow_lidar[:,1], color='blue', linewidth=2)
    plt.xlim(-15,25)
    plt.ylim(-18,22)
    plt.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False, labeltop=False, labelright=False, labelbottom=False)
    plt.savefig(os.path.join('figure3-plots', '3d-groundtruth.png'), bbox_inches='tight')
