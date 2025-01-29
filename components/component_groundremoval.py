import numpy as np
import os
import time
import torch
from nuscenes.nuscenes import NuScenes
from sklearn.linear_model import RANSACRegressor
from typing import Dict, List, Tuple
from utils.utils_functions import get_lidar_sweep, get_T_plane_reference



def ground_point_removal(pc_lidar:torch.Tensor, hyperparameters:Dict) -> Tuple[torch.Tensor,torch.Tensor]:
    """
    Remove ground points of a LiDAR point cloud.
    
    Args:
        pc_lidar (torch.Tensor) : LiDAR point cloud expressed in LiDAR frame with shape (N,4).
        hyperparameters (dict) : Hyperparameters for ground removal.
    
    Returns:
        boolall_ground (torch.Tensor) : Tensor indicating for each point whether it is ground (True) or non-ground (False) with shape (N).
        Ts_coneplane_lidar (torch.Tensor) : Tensor containing homogeneous transformation matrices for mapping 3D points from reference frame to plane frame with shape (num_cones+1,4,4). Last matrix is global plane.
    """
    
    
    # Ground Point Removal: Step 1 (extract disk that contains most of ground points).
    xyradius_thres = hyperparameters['Step1__xyradius_threshold']   # Unit: meters.
    zmin_thres     = hyperparameters['Step1__zmin_threshold']   # Unit: meters.
    zmax_thres     = hyperparameters['Step1__zmax_threshold']   # Unit: meters.
    
    boolall_xy  = torch.linalg.norm(pc_lidar[:,:2], ord=2, axis=1)<=xyradius_thres
    boolall_z   = (pc_lidar[:,2]>=zmin_thres) & (pc_lidar[:,2]<=zmax_thres)
    boolall_xyz = boolall_xy & boolall_z
    
    grounddisk_pc_lidar = pc_lidar[boolall_xyz].clone()
    
    
    # Ground Point Removal: Step 2 (fit global plane using RANSAC; plane equation is a*x+b*y+c*z+d=0).
    number_disk_points = grounddisk_pc_lidar.shape[0]
    ransac             = RANSACRegressor(min_samples=min(number_disk_points,hyperparameters['Step2__min_sample_points']), residual_threshold=hyperparameters['Step2__residual_threshold'], max_trials=hyperparameters['Step2__max_trials'], random_state=0)
    ransac.fit(grounddisk_pc_lidar[:,[0,1]].clone(), grounddisk_pc_lidar[:,2].clone())
    
    plane_parameters    = torch.zeros([4])   # Parameters [a,b,c,d].
    plane_parameters[0] = float(ransac.estimator_.coef_[0])
    plane_parameters[1] = float(ransac.estimator_.coef_[1])
    plane_parameters[2] = -1
    plane_parameters[3] = float(ransac.estimator_.intercept_)
    plane_parameters    = -plane_parameters/torch.linalg.norm(plane_parameters[:3])   # Normalize vector <a,b,c>.
    
    T_globalplane_lidar = get_T_plane_reference(plane_parameters)
    globalplane_height  = ransac.estimator_.intercept_
    
    
    # Ground Point Removal: Step 3 (divide point cloud into cone-shaped regions).
    dmax_thres = hyperparameters['Step3__dmax_thres']   # Unit: meters.
    num_cones  = hyperparameters['Step3__num_cones']   # Unit: 1.
    
    fov_cone   = 360/num_cones   # Unit: degrees.
    
    pc_globalplane            = (T_globalplane_lidar @ pc_lidar.clone().T).T
    grounddisk_pc_globalplane = (T_globalplane_lidar @ grounddisk_pc_lidar.clone().T).T
    
    boolall_ground            = torch.zeros([pc_lidar.shape[0]], dtype=torch.bool)
    angall_globalplane        = (180/np.pi*torch.arctan2(pc_globalplane[:,1], pc_globalplane[:,0])+360)%360
    anggrounddisk_globalplane = (180/np.pi*torch.arctan2(grounddisk_pc_globalplane[:,1], grounddisk_pc_globalplane[:,0])+360)%360
    Ts_coneplane_lidar        = []
    for cone_idx in range(num_cones):
        boolall_ang        = (angall_globalplane>=cone_idx*fov_cone) & (angall_globalplane<(cone_idx+1)*fov_cone)
        boolgrounddisk_ang = (anggrounddisk_globalplane>=cone_idx*fov_cone) & (anggrounddisk_globalplane<(cone_idx+1)*fov_cone)
        
        groundcone_pc_lidar = grounddisk_pc_lidar[boolgrounddisk_ang].clone()
        
        number_cone_points = groundcone_pc_lidar.shape[0]
        if number_cone_points>=hyperparameters['Step3__min_number_cone_points']:
            ransac = RANSACRegressor(min_samples=min(number_cone_points,hyperparameters['Step3__min_sample_points']), residual_threshold=hyperparameters['Step3__residual_threshold'], max_trials=hyperparameters['Step3__max_trials'], random_state=0)
            ransac.fit(groundcone_pc_lidar[:,[0,1]].clone(), groundcone_pc_lidar[:,2].clone())
            
            plane_parameters    = torch.zeros([4])   # Parameters [a,b,c,d].
            plane_parameters[0] = float(ransac.estimator_.coef_[0])
            plane_parameters[1] = float(ransac.estimator_.coef_[1])
            plane_parameters[2] = -1
            plane_parameters[3] = float(ransac.estimator_.intercept_)
            plane_parameters    = -plane_parameters/np.linalg.norm(plane_parameters[:3])   # Normalize vector <a,b,c>.
            
            T_coneplane_lidar = get_T_plane_reference(plane_parameters)
            
            if ransac.estimator_.intercept_-globalplane_height<=0.15:
                pc_coneplane = (T_coneplane_lidar @ pc_lidar.clone().T).T
                Ts_coneplane_lidar.append(T_coneplane_lidar)
            else:
                # Use global plane if cone plane is relatively high at LiDAR origin.
                pc_coneplane = (T_globalplane_lidar @ pc_lidar.clone().T).T
                Ts_coneplane_lidar.append(T_globalplane_lidar)
        else:
            # Use global plane if cone has not enough points.
            pc_coneplane = (T_globalplane_lidar @ pc_lidar.clone().T).T
            Ts_coneplane_lidar.append(T_globalplane_lidar)
            
        boolall_coneground                 = boolall_ang & (pc_coneplane[:,2]<=dmax_thres)
        boolall_ground[boolall_coneground] = True
        
    Ts_coneplane_lidar.append(T_globalplane_lidar)
    Ts_coneplane_lidar = torch.stack(Ts_coneplane_lidar)
    
    
    return boolall_ground, Ts_coneplane_lidar



def main__ground_point_removal(nusc:NuScenes, scenes:List, hyperparameters:Dict, intermediate_results_groundremoval_dir1:str, intermediate_results_groundremoval_dir2:str, first_scene:int=0, num_of_scenes:int=850):
    """
    Remove ground points for all LiDAR sweeps.
    
    Args:
        nusc (NuScenes) : NuScenes Object.
        scenes (list) : List with scene dictionaries of dataset.
        hyperparameters (dict) : Hyperparameters for ground removal.
        intermediate_results_groundremoval_dir1 (str) : Folder for ground removal results (boolean arrays).
        intermediate_results_groundremoval_dir2 (str) : Folder for ground removal results (Ts_coneplane_lidar).
        first_scene (int) : index of first scene for removing ground points.
        num_of_scenes (int) : Number of scenes for removing ground points.
    """
    
    
    for scene_idx in range(min(first_scene,len(scenes)), min(first_scene+num_of_scenes,len(scenes))):
        print(f'--- scene_idx: {scene_idx}')
        
        
        for sweep_idx in range(len(scenes[scene_idx]['sweep_lidar_tokens'])):
            
            
            # Get sweep information.
            lidar_token          = scenes[scene_idx]['sweep_lidar_tokens'][sweep_idx]
            T_mainvehicle_global = torch.linalg.inv(scenes[scene_idx]['sweep_lidar_T_global_vehicle'][sweep_idx])
            T_vehicle_lidar      = scenes[scene_idx]['T_vehicle_lidar']
            
            
            # Get LiDAR points.
            pc_lidar        = get_lidar_sweep(nusc=nusc, lidar_token=lidar_token)
            original_length = pc_lidar.shape[0]
            
            
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
            
            extra_pc_mainlidar_list = []
            for idx in extra_ids:
                extra_pc_lidar       = get_lidar_sweep(nusc=nusc, lidar_token=scenes[scene_idx]['sweep_lidar_tokens'][idx])
                extra_T_global_lidar = scenes[scene_idx]['sweep_lidar_T_global_vehicle'][idx] @ T_vehicle_lidar
                extra_pc_mainlidar   = (torch.linalg.inv(T_vehicle_lidar) @ T_mainvehicle_global @ extra_T_global_lidar @ extra_pc_lidar.T).T
                extra_pc_mainlidar_list.append(extra_pc_mainlidar.clone())
            if len(extra_ids)>0:
                pc_lidar = torch.concatenate((pc_lidar, torch.concatenate(extra_pc_mainlidar_list)), dim=0)
                
                
            # Remove ground.
            boolall_ground, Ts_coneplane_lidar = ground_point_removal(pc_lidar=pc_lidar.clone(), hyperparameters=hyperparameters)
            boolall_ground = boolall_ground[:original_length].int()
            
            
            # Save boolean array.
            lidar_token    = scenes[scene_idx]['sweep_lidar_tokens'][sweep_idx]
            lidar_record   = nusc.get('sample_data', lidar_token)
            lidar_filename = os.path.basename(lidar_record['filename']).replace('.pcd.bin','.npy')
            np.save(os.path.join(intermediate_results_groundremoval_dir1, lidar_filename), boolall_ground.numpy())
            
            
            # Save Ts_coneplane_lidar.
            lidar_token    = scenes[scene_idx]['sweep_lidar_tokens'][sweep_idx]
            lidar_record   = nusc.get('sample_data', lidar_token)
            lidar_filename = os.path.basename(lidar_record['filename']).replace('.pcd.bin','.npy')
            np.save(os.path.join(intermediate_results_groundremoval_dir2, lidar_filename), Ts_coneplane_lidar.numpy())
