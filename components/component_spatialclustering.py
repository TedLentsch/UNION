import hdbscan
import numpy as np
import os
import torch
import types
from nuscenes.nuscenes import NuScenes
from typing import Dict, List, Tuple
from utils.utils_functions import get_lidar_sweep



# Bounding box fitting similar as MODEST (closeness_rectangle function)
# Original version from https://github.com/YurongYou/MODEST/blob/master/generate_cluster_mask/utils/pointcloud_utils.py.
# Rewrote function, changed output of function, and added some comments.
def fit_bounding_box(cluster_reference:torch.Tensor) -> Tuple[torch.Tensor, List]:
    """
    Fit BEV bounding box around cluster points.
    
    Args:
        cluster_reference (torch.Tensor) : LiDAR point cloud of cluster expressed in reference frame with shape (N,4).
    
    Returns:
        T_reference_bbox (torch.Tensor) : Homogeneous transformation matrix for mapping 3D points from bbox frame to reference frame.
        bboxdimensions (list) : List with bounding box dimensions [length, width, and height].
    """
    
    
    # Map points to local frame using a rotation around reference z-axis and select rotation based on closeness metric beta.
    delta        = 1
    d0           = torch.tensor(1e-2)
    max_beta     = -float('inf')
    choose_angle = None
    for angle in torch.arange(0, 90+delta, delta):
        angle = 180/np.pi*angle
        
        R_local_reference = torch.tensor([[ torch.cos(angle), torch.sin(angle)],
                                          [-torch.sin(angle), torch.cos(angle)]])
        
        cluster_local = (R_local_reference @ cluster_reference[:,:2].T).T
        
        min_x, max_x = cluster_local[:,0].min(), cluster_local[:,0].max()
        min_y, max_y = cluster_local[:,1].min(), cluster_local[:,1].max()
        Dx           = torch.vstack((cluster_local[:,0]-min_x, max_x-cluster_local[:,0])).min(dim=0).values
        Dy           = torch.vstack((cluster_local[:,1]-min_y, max_y-cluster_local[:,1])).min(dim=0).values
        
        beta = torch.vstack((Dx, Dy)).min(dim=0).values
        beta = torch.maximum(beta, d0)
        beta = 1/beta
        beta = beta.sum()
        
        if beta>max_beta:
            max_beta     = beta
            choose_angle = angle
            
            
    # Get minimum and maximum x and y values in local frame.
    angle = choose_angle
    
    R_local_reference = torch.tensor([[ torch.cos(angle), torch.sin(angle)],
                                      [-torch.sin(angle), torch.cos(angle)]])
    
    cluster_local = (R_local_reference @ cluster_reference[:,:2].T).T
    
    min_x, max_x = cluster_local[:,0].min(), cluster_local[:,0].max()
    min_y, max_y = cluster_local[:,1].min(), cluster_local[:,1].max()
    
    
    # X-axis is aligned with longest 2D dimension.
    if (max_x-min_x)<(max_y-min_y):
        angle = choose_angle+np.pi/2
        
        R_local_reference = torch.tensor([[ torch.cos(angle), torch.sin(angle)],
                                          [-torch.sin(angle), torch.cos(angle)]])
        
        cluster_local = (R_local_reference @ cluster_reference[:,:2].T).T
        
        min_x, max_x = cluster_local[:,0].min(), cluster_local[:,0].max()
        min_y, max_y = cluster_local[:,1].min(), cluster_local[:,1].max()
        
        
    # Calculate corners of 2D bounding box.
    corners_local = torch.tensor([[max_x, min_y],
                                  [min_x, min_y],
                                  [min_x, max_y],
                                  [max_x, max_y]])
    corners_reference = (R_local_reference.T @ corners_local.T).T
    
    
    # Calculate bounding box center (bottom).
    bboxcenter_reference    = torch.zeros([3])
    bboxcenter_reference[0] = corners_reference[:,0].sum()/4
    bboxcenter_reference[1] = corners_reference[:,1].sum()/4
    bboxcenter_reference[2] = cluster_reference[:,2].min()
    
    
    # Calculate T_reference_cluster.
    T_reference_bbox       = torch.eye(4)
    T_reference_bbox[:3,3] = bboxcenter_reference
    T_reference_bbox[:2,0] = R_local_reference.T[:,0]
    T_reference_bbox[:2,1] = R_local_reference.T[:,1]
    
    
    # Calculate bounding box dimensions.
    bboxlength     = torch.linalg.norm(corners_reference[1]-corners_reference[0])
    bboxwidth      = torch.linalg.norm(corners_reference[-1]-corners_reference[0])
    bboxheight     = cluster_reference[:,2].max()-cluster_reference[:,2].min()
    bboxdimensions = [bboxlength.item(), bboxwidth.item(), bboxheight.item()]
    
    
    return T_reference_bbox, bboxdimensions



def spatial_clustering(pc_lidar:torch.Tensor, boolall_ground:torch.Tensor, Ts_coneplane_lidar:torch.Tensor, original_lengths:Dict, hyperparameters:Dict) -> Dict:
    """
    Cluster inlier points of point cloud spatially.
    
    Args:
        pc_lidar (torch.Tensor) : LiDAR point cloud expressed in LiDAR frame with shape (N,4).
        boolall_ground (torch.Tensor) : Boolean tensor indicating whether point is ground with shape (N,4).
        Ts_coneplane_lidar (torch.Tensor) : Tensor containing homogeneous transformation matrices for mapping 3D points from reference frame to plane frame with shape (num_cones,4,4).
        original_lengths (dict) : Original number of points in each sweep (before aggregation).
        hyperparameters (dict) : Hyperparameters for spatial clustering.
    
    Returns:
        cluster_dict (dict) : Dictionary containing indices of cluster and fitted bounding box.
    """
    
    
    # Spatial Clustering: Step 0 (get M).
    M = hyperparameters['Step0__M']   # Unit: 1.
    
    
    # Spatial Clustering: Step 1 (get inlier points by removing ground points, sky points, and far away points).
    sky_thres     = hyperparameters['Step1__sky_threshold']   # Unit: meters.
    range_thres   = hyperparameters['Step1__range_threshold']   # Unit: meters.
    x_range_thres = hyperparameters['Step1__x_range_threshold']   # Unit: meters.
    y_range_thres = hyperparameters['Step1__y_range_threshold']   # Unit: meters.
    
    T_globalplane_lidar = Ts_coneplane_lidar[0]
    boolall_sky         = (T_globalplane_lidar @ pc_lidar.T).T[:,2]>=sky_thres
    if range_thres is not None:
        boolall_outrange = torch.linalg.norm(pc_lidar[:,:2], ord=2, axis=1)>range_thres
    else:
        boolall_outrange = (abs(pc_lidar[:,0])>x_range_thres) | (abs(pc_lidar[:,1])>y_range_thres)
    boolall_outlier = boolall_ground | boolall_sky | boolall_outrange
    idsall_inlier   = torch.where(~boolall_outlier)[0]
    
    inlier_pc_lidar = pc_lidar[idsall_inlier].clone()
    
    
    # Spatial Clustering: Step 2 (cluster inlier points using HDBSCAN).
    clustersize_thres         = hyperparameters['Step2__clustersize_threshold']   # Unit: 1.
    cluster_selection_epsilon = hyperparameters['Step2__cluster_selection_epsilon']   # Unit: 1.
    
    total_num_frames = len(original_lengths)
    
    hdbscan_clusterer      = hdbscan.HDBSCAN(min_cluster_size=clustersize_thres, metric='euclidean', cluster_selection_epsilon=cluster_selection_epsilon)
    hdbscan_cluster_labels = torch.IntTensor(hdbscan_clusterer.fit_predict(inlier_pc_lidar[:,:3]))
    
    
    # Spatial Clustering: Step 3 (fit bounding box using same method as MODEST).
    num_cones = hyperparameters['Step3__num_cones']   # Unit: 1.
    
    fov_cone = 360/num_cones   # Unit: degrees.
    
    timestamp_tensor = torch.concatenate([key*torch.ones([original_lengths[key][1]-original_lengths[key][0]]) for key in list(original_lengths.keys())], dim=0)
    
    cluster_dict = {}
    for label in torch.unique(hdbscan_cluster_labels)[1:].tolist():
        # Get cluster indices (HDBSCAN output).
        idsall_cluster = idsall_inlier[torch.where(hdbscan_cluster_labels==label)[0]]
        
        # Fit bounding box.
        T_lidar_bbox, bboxdimensions = fit_bounding_box(pc_lidar.clone()[idsall_cluster,:])
        
        # Get cluster indices (inside fitted bounding box).
        inlier_pc_bbox     = (torch.linalg.inv(T_lidar_bbox) @ inlier_pc_lidar.clone().T).T
        boolall_insidebbox = (torch.abs(inlier_pc_bbox[:,0])<=bboxdimensions[0]/2) & (torch.abs(inlier_pc_bbox[:,1])<=bboxdimensions[1]/2) & (inlier_pc_bbox[:,2]>=0) & (inlier_pc_bbox[:,2]<=bboxdimensions[2])
        idsall_insidebbox  = idsall_inlier[boolall_insidebbox]
        
        # Determine height of bounding box center above ground (can be negative).
        cone_idx            = int((((180/np.pi*torch.arctan2(T_lidar_bbox[1,3], T_lidar_bbox[0,3])+360)%360)/fov_cone)%num_cones)
        T_coneplane_lidar   = Ts_coneplane_lidar[cone_idx]
        height_above_ground = (T_coneplane_lidar @ T_lidar_bbox)[2,3].item()
        
        # Determine bounding box that touches ground.
        touchground_T_lidar_bbox       = T_lidar_bbox.clone()
        touchground_T_lidar_bbox[2,3] += -height_above_ground
        touchground_bboxdimensions     = bboxdimensions.copy()
        touchground_bboxdimensions[2] += height_above_ground
        
        # Create cluster.
        cluster = types.SimpleNamespace()
        cluster.avg_number_points          = len(idsall_cluster)/total_num_frames
        cluster.T_lidar_bbox               = T_lidar_bbox.numpy()
        cluster.bboxdimensions             = bboxdimensions
        cluster.yaw_radians                = torch.arctan2(T_lidar_bbox[1,0], T_lidar_bbox[0,0]).item()
        cluster.touchground_T_lidar_bbox   = touchground_T_lidar_bbox.numpy()
        cluster.touchground_bboxdimensions = touchground_bboxdimensions
        cluster.touchground_yaw_radians    = torch.arctan2(touchground_T_lidar_bbox[1,0], touchground_T_lidar_bbox[0,0]).item()
        cluster.height_above_ground        = height_above_ground
        cluster.idsall_aggregated          = idsall_cluster.tolist()
        cluster.idsall_frame               = {key:idsall_cluster[timestamp_tensor[idsall_cluster]==key].tolist() for key in list(original_lengths.keys())}
        cluster.idsall_aggregated2         = idsall_insidebbox.tolist()
        cluster.idsall_frame2              = {key:idsall_insidebbox[timestamp_tensor[idsall_insidebbox]==key].tolist() for key in list(original_lengths.keys())}
        
        # Store cluster information.
        cluster_dict[label] = cluster
        
        
    # Spatial Clusterinig: Step 4 (filter clusters based on bounding box size and position).
    length_max_threshold              = hyperparameters['Step4__length_max_threshold']   # Unit: meters.
    width_max_threshold               = hyperparameters['Step4__width_max_threshold']   # Unit: meters.
    height_min_threshold              = hyperparameters['Step4__height_min_threshold']   # Unit: meters.
    height_above_ground_max_threshold = hyperparameters['Step4__height_above_ground_max_threshold']   # Unit: meters.
    length_width_max_ratio_threshold  = hyperparameters['Step4__length_width_max_ratio_threshold']   # Unit: 1.
    area_min_threshold                = hyperparameters['Step4__area_min_threshold']   # Unit: square meters.
    
    sorted_cluster_list = [v for k, v in sorted(cluster_dict.items(), key=lambda item: item[1].avg_number_points)]
    filtered_cluster_list = []
    for cluster in sorted_cluster_list:
        bboxlength, bboxwidth, bboxheight = cluster.bboxdimensions
        height_above_ground               = cluster.height_above_ground
        
        if bboxlength>length_max_threshold:
            continue
        elif bboxwidth>width_max_threshold:
            continue
        elif bboxheight<height_min_threshold:
            continue
        elif height_above_ground>height_above_ground_max_threshold:
            continue
        elif bboxlength/bboxwidth>length_width_max_ratio_threshold:
            continue
        elif bboxlength*bboxwidth<area_min_threshold:
            continue
        else:
            # At least 1 point in current sweep.
            if len(cluster.idsall_frame[0])>0:
                filtered_cluster_list.append(cluster)
    cluster_dict = dict(zip(range(len(filtered_cluster_list)), filtered_cluster_list))
    
    
    return cluster_dict



def main__spatial_clustering(nusc:NuScenes, scenes:List, hyperparameters:Dict, intermediate_results_groundremoval_dir1:str, intermediate_results_groundremoval_dir2:str, intermediate_results_spatialclustering_dir:str, first_scene:int=0, num_of_scenes:int=850):
    """
    Cluster points for all samples.
    
    Args:
        nusc (NuScenes) : NuScenes Object.
        scenes (list) : List with scene dictionaries of dataset.
        hyperparameters (dict) : Hyperparameters for spatial clustering.
        intermediate_results_groundremoval_dir1 (str) : Folder for ground removal results (boolean arrays).
        intermediate_results_groundremoval_dir2 (str) : Folder for ground removal results (Ts_coneplane_lidar).
        intermediate_results_spatialclustering_dir (str) : Folder for spatial clustering results (cluster dicts).
        first_scene (int) : Index of first scene for removing ground points.
        num_of_scenes (int) : Number of scenes for removing ground points.
    """
    
    
    for scene_idx in range(min(first_scene,len(scenes)), min(first_scene+num_of_scenes,len(scenes))):
        print(f'--- scene_idx: {scene_idx}')
        
        
        for sample_idx in range(len(scenes[scene_idx]['sample_tokens'])):
            
            
            # Get sweep index.
            timestamp = scenes[scene_idx]['sample_timestamps'][sample_idx]
            sweep_idx = scenes[scene_idx]['sweep_lidar_timestamps'].index(timestamp)
            
            
            # Get sweep information.
            lidar_token          = scenes[scene_idx]['sample_lidar_tokens'][sample_idx]
            T_mainvehicle_global = torch.linalg.inv(scenes[scene_idx]['sweep_lidar_T_global_vehicle'][sweep_idx])
            T_vehicle_lidar      = scenes[scene_idx]['T_vehicle_lidar']
            
            
            # Get LiDAR points.
            pc_lidar         = get_lidar_sweep(nusc=nusc, lidar_token=lidar_token)
            original_lengths = {0: [0,pc_lidar.shape[0]],}
            
            
            # Get ground segmentation.
            lidar_record       = nusc.get('sample_data', lidar_token)
            filename           = os.path.basename(lidar_record['filename']).replace('.pcd.bin','.npy')
            boolall_ground     = torch.from_numpy(np.load(os.path.join(intermediate_results_groundremoval_dir1, filename)))>0
            Ts_coneplane_lidar = torch.from_numpy(np.load(os.path.join(intermediate_results_groundremoval_dir2, filename)))
            
            
            # Add more frames [-M,+M]; this results in 2*M+1 frames.
            M = hyperparameters['Step0__M']   # Unit: 1.
            
            sweep_ids = list(range(len(scenes[scene_idx]['sweep_lidar_tokens'])))
            
            past_extra_ids = list(range(max(sweep_ids.index(sweep_idx)-M,0), sweep_ids.index(sweep_idx)))
            next_extra_ids = list(range(min(sweep_ids.index(sweep_idx)+1,len(sweep_ids)), min(sweep_ids.index(sweep_idx)+1+M,len(sweep_ids))))
            if len(past_extra_ids)<M:
                next_extra_ids.extend(list(range(next_extra_ids[-1]+1, next_extra_ids[-1]+1+M-len(past_extra_ids))))
            elif len(next_extra_ids)<M:
                past_extra_ids.extend(list(range(past_extra_ids[0]-(M-len(next_extra_ids)), past_extra_ids[0])))
            extra_ids = torch.sort(torch.IntTensor(past_extra_ids+next_extra_ids)).values.tolist()
            extra_ids = torch.IntTensor(sweep_ids)[extra_ids].tolist()
            
            extra_pc_mainlidar_list   = []
            extra_boolall_ground_list = []
            for idx in extra_ids:
                extra_pc_lidar       = get_lidar_sweep(nusc=nusc, lidar_token=scenes[scene_idx]['sweep_lidar_tokens'][idx])
                extra_T_global_lidar = scenes[scene_idx]['sweep_lidar_T_global_vehicle'][idx] @ T_vehicle_lidar
                extra_pc_mainlidar   = (torch.linalg.inv(T_vehicle_lidar) @ T_mainvehicle_global @ extra_T_global_lidar @ extra_pc_lidar.T).T
                extra_pc_mainlidar_list.append(extra_pc_mainlidar.clone())
                original_lengths[idx-sweep_idx] = [original_lengths[list(original_lengths.keys())[-1]][-1],original_lengths[list(original_lengths.keys())[-1]][-1]+extra_pc_lidar.shape[0]]
                lidar_record          = nusc.get('sample_data', scenes[scene_idx]['sweep_lidar_tokens'][idx])
                segmentation_filename = os.path.basename(lidar_record['filename']).replace('.pcd.bin','.npy')
                extra_boolall_ground  = torch.from_numpy(np.load(os.path.join(intermediate_results_groundremoval_dir1, segmentation_filename)))>0
                extra_boolall_ground_list.append(extra_boolall_ground.clone())
            if len(extra_ids)>0:
                pc_lidar       = torch.concatenate((pc_lidar, torch.concatenate(extra_pc_mainlidar_list)), dim=0)
                boolall_ground = torch.concatenate((boolall_ground, torch.concatenate(extra_boolall_ground_list)), dim=0)
                
                
            # Get cluster dict.
            cluster_dict = spatial_clustering(pc_lidar=pc_lidar, boolall_ground=boolall_ground, Ts_coneplane_lidar=Ts_coneplane_lidar, original_lengths=original_lengths, hyperparameters=hyperparameters)
            
            
            # Save dictionary (everything).
            lidar_token  = scenes[scene_idx]['sample_lidar_tokens'][sample_idx]
            lidar_record = nusc.get('sample_data', lidar_token)
            filename     = os.path.basename(lidar_record['filename']).replace('.pcd.bin','.npy')
            np.save(os.path.join(intermediate_results_spatialclustering_dir, filename), cluster_dict)
            
            
            # Save disctionary (drop point indices; for fast loading).
            for cluster_idx in cluster_dict.keys():
                cluster_dict[cluster_idx].idsall_aggregated = None
                cluster_dict[cluster_idx].idsall_frame = None
                cluster_dict[cluster_idx].idsall_aggregated2 = None
                cluster_dict[cluster_idx].idsall_frame2 = None
            lidar_token  = scenes[scene_idx]['sample_lidar_tokens'][sample_idx]
            lidar_record = nusc.get('sample_data', lidar_token)
            filename     = os.path.basename(lidar_record['filename']).replace('.pcd.bin','__small.npy')
            np.save(os.path.join(intermediate_results_spatialclustering_dir, filename), cluster_dict)
