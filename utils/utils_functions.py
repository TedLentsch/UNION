import numpy as np
import os
import torch
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion
from typing import List, Tuple



def get_scene_information(nusc:NuScenes) -> List:
    """
    Extract scene information using NuScenes object and store it in a single list for easier use.
    
    Args:
        nusc (nuscenes.nuscenes.NuScenes) : NuScenes object.
        
    Return:
        scenes (list) : Information of all scenes stored in a single list.
    """
    
    
    scenes = []
    for scene_idx in range(len(nusc.scene)):
        scene_dict                                  = {}
        scene_dict['scene_idx']                     = scene_idx
        scene_dict['scene_name']                    = nusc.scene[scene_idx]['name']
        scene_dict['scene_description']             = nusc.scene[scene_idx]['description']
        scene_dict['scene_token']                   = nusc.scene[scene_idx]['token']
        
        scene_dict['number_of_samples']             = nusc.scene[scene_idx]['nbr_samples']
        
        scene_dict['sample_tokens']                 = [None for _ in range(nusc.scene[scene_idx]['nbr_samples'])]
        scene_dict['sample_timestamps']             = [None for _ in range(nusc.scene[scene_idx]['nbr_samples'])]
        scene_dict['sample_lidar_tokens']           = [None for _ in range(nusc.scene[scene_idx]['nbr_samples'])]
        scene_dict['sample_lidar_T_global_vehicle'] = [None for _ in range(nusc.scene[scene_idx]['nbr_samples'])]
        scene_dict['sample_annotations']            = [None for _ in range(nusc.scene[scene_idx]['nbr_samples'])]
        
        scene_dict['sweep_lidar_timestamps']        = []
        scene_dict['sweep_lidar_tokens']            = []
        scene_dict['sweep_lidar_T_global_vehicle']  = []
        
        scene_dict['T_vehicle_lidar']               = [None]
        
        
        # Add sample related information.
        scene_dict['sample_tokens'][0] = nusc.scene[scene_idx]['first_sample_token']
        for sample_idx in range(nusc.scene[scene_idx]['nbr_samples']-1):
            sample_token                                            = scene_dict['sample_tokens'][sample_idx]
            sample_record                                           = nusc.get('sample', sample_token)
            sample_lidar_record                                     = nusc.get('sample_data', sample_record['data']['LIDAR_TOP'])
            sample_lidar_egopose_record                             = nusc.get('ego_pose', sample_lidar_record['ego_pose_token'])
            scene_dict['sample_timestamps'][sample_idx]             = sample_record['timestamp']
            scene_dict['sample_lidar_tokens'][sample_idx]           = sample_record['data']['LIDAR_TOP']
            scene_dict['sample_lidar_T_global_vehicle'][sample_idx] = torch.from_numpy(transform_matrix(sample_lidar_egopose_record['translation'], Quaternion(sample_lidar_egopose_record['rotation']), inverse=False)).float()
            scene_dict['sample_annotations'][sample_idx]            = sample_record['anns']
            scene_dict['sample_tokens'][sample_idx+1]               = sample_record['next']
        sample_token                                    = scene_dict['sample_tokens'][-1]
        sample_record                                   = nusc.get('sample', sample_token)
        sample_lidar_record                             = nusc.get('sample_data', sample_record['data']['LIDAR_TOP'])
        sample_lidar_egopose_record                     = nusc.get('ego_pose', sample_lidar_record['ego_pose_token'])
        scene_dict['sample_timestamps'][-1]             = sample_record['timestamp']
        scene_dict['sample_lidar_tokens'][-1]           = sample_record['data']['LIDAR_TOP']
        scene_dict['sample_lidar_T_global_vehicle'][-1] = torch.from_numpy(transform_matrix(sample_lidar_egopose_record['translation'], Quaternion(sample_lidar_egopose_record['rotation']), inverse=False)).float()
        scene_dict['sample_annotations'][-1]            = sample_record['anns']
        
        
        # Add LiDAR sweep related information.
        first_sample_token  = scene_dict['sample_tokens'][0]
        first_sample_record = nusc.get('sample', first_sample_token)
        scene_dict['sweep_lidar_tokens'].append(first_sample_record['data']['LIDAR_TOP'])
        
        sweep_idx = 0
        while True:
            sweep_lidar_record         = nusc.get('sample_data', scene_dict['sweep_lidar_tokens'][sweep_idx])
            sweep_lidar_egopose_record = nusc.get('ego_pose', sweep_lidar_record['ego_pose_token'])
            scene_dict['sweep_lidar_timestamps'].append(sweep_lidar_record['timestamp'])
            scene_dict['sweep_lidar_T_global_vehicle'].append(torch.from_numpy(transform_matrix(sweep_lidar_egopose_record['translation'], Quaternion(sweep_lidar_egopose_record['rotation']), inverse=False)).float())
            if sweep_lidar_record['next']!='':
                scene_dict['sweep_lidar_tokens'].append(sweep_lidar_record['next'])
                sweep_idx += 1
            else:
                break
                
                
        # Add transform_vehicle_lidar.
        first_sample_token            = scene_dict['sample_tokens'][0]
        first_sample_record           = nusc.get('sample', first_sample_token)
        first_sample_lidar_record     = nusc.get('sample_data', first_sample_record['data']['LIDAR_TOP'])
        lidar_sensorpose_token        = first_sample_lidar_record['calibrated_sensor_token']
        lidar_sensorpose_record       = nusc.get('calibrated_sensor', lidar_sensorpose_token)
        scene_dict['T_vehicle_lidar'] = torch.from_numpy(transform_matrix(lidar_sensorpose_record['translation'], Quaternion(lidar_sensorpose_record['rotation']), inverse=False)).float()
        
        
        # Append to list.
        scenes.append(scene_dict)
        
        
    return scenes



def get_lidar_sweep(nusc:NuScenes, lidar_token:str, radius:float=2.5) -> torch.Tensor:
    """
    Loads LiDAR points, removes points close to sensor, and creates point cloud with shape (N,4).
    
    Args:
        nusc (nuscenes.nuscenes.NuScenes) : NuScenes object.
        lidar_token (str) : LiDAR token.
        radius (float) : Threshold for removing points close to sensor (meters).
        
    Return:
        sweep_pc_lidar (torch.Tensor) : LiDAR point cloud expressed in LiDAR frame with shape (N,4).
    """
    
    
    # Get LiDAR record and remove points close to sensor.
    lidar_record    = nusc.get('sample_data', lidar_token)
    sweep_obj_lidar = LidarPointCloud.from_file(os.path.join(nusc.dataroot, lidar_record['filename']))
    sweep_obj_lidar.remove_close(radius=radius)
    
    
    # Get LiDAR points and append ones.
    sweep_pc_lidar = torch.from_numpy(sweep_obj_lidar.points[:3,:]).T
    sweep_pc_lidar = torch.concatenate((sweep_pc_lidar[:,:3], torch.ones([sweep_pc_lidar.shape[0],1], dtype=torch.float32)), dim=1)
    
    
    return sweep_pc_lidar



def get_T_plane_reference(plane_parameters:torch.Tensor) -> torch.Tensor:
    """
    Get homogeneous transformation matrix T_plane_reference.
    
    Args:
        plane_parameters (torch.Tensor) : Plane parameters for plane equation a*x+b*y+c*z+d=0.
        
    Returns:
        T_plane_reference (torch.Tensor) : Homogeneous transformation matrix for mapping 3D points from reference frame to plane frame.
    """
    
    
    T_reference_plane       = torch.eye(4)
    reference_P_plane_org   = torch.Tensor([0, 0, -plane_parameters[3]/plane_parameters[2], 1])
    reference_X_plane_vec   = torch.Tensor([1, 0, -(1*plane_parameters[0]+plane_parameters[3])/plane_parameters[2]])-reference_P_plane_org[:3]
    reference_X_plane_vec   = reference_X_plane_vec/torch.linalg.norm(reference_X_plane_vec)
    reference_Z_plane_vec   = plane_parameters[:3]    
    reference_Y_plane_vec   = torch.cross(reference_Z_plane_vec, reference_X_plane_vec)
    T_reference_plane[:3,0] = reference_X_plane_vec
    T_reference_plane[:3,1] = reference_Y_plane_vec
    T_reference_plane[:3,2] = reference_Z_plane_vec
    T_reference_plane[:4,3] = reference_P_plane_org
    T_plane_reference       = torch.linalg.inv(T_reference_plane)
    
    
    return T_plane_reference
