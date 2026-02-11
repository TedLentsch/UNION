import cv2
import numpy as np
import os
import torch
import torchvision
import types
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import transform_matrix
from pathlib import Path
from pyquaternion import Quaternion
from typing import List, Dict, Tuple
from utils.utils_functions import get_lidar_sweep



class ToTensor():
    """ Convert image to tensors with type float. """
    
    def __call__(self, img:np.ndarray,) -> torch.Tensor:
        img = torchvision.transforms.functional.to_tensor(img.copy()).float()
                
        return img



class NormalizeTensor():
    """ Normalize tensors by using PyTorch ImageNet mean and standard deviation (std). """
    
    def __init__(self):
        self.mean = [0.485, 0.456, 0.406]
        self.std  = [0.229, 0.224, 0.225]
        
    def __call__(self, img:torch.Tensor) -> torch.Tensor:
        img = torchvision.transforms.functional.normalize(img, self.mean, self.std)
        
        return img



class Compose():
    """ Apply all transformations. """
    
    def __init__(self, transforms:List,):
        self.transforms = transforms
        
    def __call__(self, img:np.ndarray,) -> torch.Tensor:
        for transform in self.transforms:
            img = transform(img)
            
        return img



def get_transform() -> Compose:
    """ Combine all transformations into a single transformation. """
    
    transforms = []
    
    # NumPy array to tensor (this scales element values from 0-255 to 0-1).
    transforms.append(ToTensor())
    
    # Normalize tensor (using ImageNet statistics).
    transforms.append(NormalizeTensor())
    
    return Compose(transforms)



def get_camera_transformations(nusc:NuScenes, sample_record:Dict, sensor_name:str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load homogeneous transformations.
    
    Args:
        nusc (NuScenes) : NuScenes Object.
        sample_record (dict) : Dictionary with information about sample.
        sensor_name (str) : Name of camera.
    
    Returns:
        T_cam_vehicle (torch.Tensor) : Homogeneous transformation matrix for mapping 3D points from camera frame to vehicle frame with shape (4,4).
        T_camvehicle_global (torch.Tensor) : Homogeneous transformation matrix for mapping 3D points from vehicle frame to global frame with shape (4,4).
        cam_proj_matrix (torch.Tensor) : Camera projection matrix of sensor.
    """
    
    
    # Camera records.
    cam_record            = nusc.get('sample_data', sample_record['data'][sensor_name])
    cam_egopose_record    = nusc.get('ego_pose', cam_record['ego_pose_token'])
    cam_sensorpose_record = nusc.get('calibrated_sensor', cam_record['calibrated_sensor_token'])
    
    
    # Camera-related transformations.
    T_camvehicle_global = transform_matrix(cam_egopose_record['translation'], Quaternion(cam_egopose_record['rotation']), inverse=True)
    T_cam_vehicle       = transform_matrix(cam_sensorpose_record['translation'], Quaternion(cam_sensorpose_record['rotation']), inverse=True)
    cam_proj_matrix     = np.hstack((cam_sensorpose_record['camera_intrinsic'], np.zeros([3,1])))
    
    
    # Move to torch.
    T_camvehicle_global = torch.from_numpy(T_camvehicle_global).float()
    T_cam_vehicle       = torch.from_numpy(T_cam_vehicle).float()
    cam_proj_matrix     = torch.from_numpy(cam_proj_matrix).float()
    
    
    return T_cam_vehicle, T_camvehicle_global, cam_proj_matrix



def get_projections_and_distance_to_center(img_cam:torch.Tensor, stride:int, pc_global:torch.Tensor, T_cam_vehicle:torch.Tensor, T_camvehicle_global:torch.Tensor, cam_proj_matrix:torch.Tensor,) -> Tuple[torch.Tensor, torch.Tensor,]:
    """
    Project LiDAR points to image plane and compute distance for each pixel to image center.
    
    Args:
        img_cam (torch.Tensor) : Camera image of sensor.
        stride (int) : Effective stride of DINOv2.
        pc_global (torch.Tensor) : LiDAR point cloud expressed in global frame with shape (N,4).
        T_cam_vehicle (torch.Tensor) : Homogeneous transformation matrix for mapping 3D points from camera frame to vehicle frame with shape (4,4).
        T_camvehicle_global (torch.Tensor) : Homogeneous transformation matrix for mapping 3D points from vehicle frame to global frame with shape (4,4).
        cam_proj_matrix (torch.Tensor) : Camera projection matrix of sensor.
        
    Returns:
        uvs (torch.Tensor) : Pixel coordinates with shape (N,2).
        dist_to_img_center (torch.Tensor) : Distance between pixel and image center in pixels with shape (N).
    """
    
    
    # Transform point cloud to camera frame.
    pc_cam = (T_cam_vehicle @ T_camvehicle_global @ pc_global.T).T
    
    
    # Project points.
    uvw = (cam_proj_matrix @ pc_cam.T).T
    uvs = torch.round(uvw[:,0:2]/uvw[:,2:3]).int()
    
    
    # Remove points outside camera FoV.
    bool_visible = pc_cam[:,2]>0   
    bool_h       = torch.logical_and(uvs[:,1]>=0, uvs[:,1]<stride*(img_cam.shape[0]//stride))
    bool_w       = torch.logical_and(uvs[:,0]>=0, uvs[:,0]<stride*(img_cam.shape[1]//stride))
    bool_hw      = torch.logical_and(bool_h, bool_w)
    
    uvs[~bool_visible] = -1
    uvs[~bool_hw]      = -1
    
    
    # Compute distance to image center.
    img_center = torch.Tensor([[stride*(img_cam.shape[0]//stride)/2, stride*(img_cam.shape[1]//stride)/2]])
    
    dist_to_img_center               = torch.linalg.norm(uvs-img_center, ord=2, axis=1)
    dist_to_img_center[uvs[:,0]==-1] = float('inf')
    
    
    return uvs, dist_to_img_center



def main__appearance_embedding(nusc:NuScenes, scenes:List, hyperparameters:Dict, intermediate_results_spatialclustering_dir:str, intermediate_results_appearanceembedding_dir:str, first_scene:int=0, num_of_scenes:int=850):
    """
    Compute appearance embedding for clusters of all samples.
    
    Args:
        nusc (NuScenes) : NuScenes Object.
        scenes (list) : List with scene dictionaries of dataset.
        hyperparameters (dict) : Hyperparameters for appearance embedding.
        intermediate_results_spatialclustering_dir (str) : Folder for spatial clustering results (cluster dicts).
        intermediate_results_appearanceembedding_dir (str) : Folder for appearance embedding results (cluster dicts).
        first_scene (int) : Index of first scene for removing ground points.
        num_of_scenes (int) : Number of scenes for removing ground points.
    """
    
    
    # Define names, get model, and get transform.
    sensor_names = ['CAM_FRONT','CAM_FRONT_RIGHT','CAM_BACK_RIGHT','CAM_BACK','CAM_BACK_LEFT','CAM_FRONT_LEFT']
    stride       = hyperparameters['Step0__stride']
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dinov2_dir = Path(__file__).resolve().parents[1] / 'dinov2'
    model  = torch.hub.load(str(dinov2_dir), 'dinov2_vitl14_reg', source='local').to(device)
    model.eval()
    
    transform = get_transform()
    
    
    # Process dataset.
    for scene_idx in range(min(first_scene,len(scenes)), min(first_scene+num_of_scenes,len(scenes))):
        print(f'--- scene_idx: {scene_idx}')
        
        
        for sample_idx in range(len(scenes[scene_idx]['sample_tokens'])):
            
            
            # Get sample record.
            sample_token  = scenes[scene_idx]['sample_tokens'][sample_idx]
            sample_record = nusc.get('sample', sample_token)
            
            
            # Compute feature maps.
            imgs_cams_list = []
            for sensor_name in sensor_names:
                cam_record = nusc.get('sample_data', sample_record['data'][sensor_name])
                img_cam    = cv2.imread(os.path.join(nusc.dataroot, cam_record['filename']))[...,::-1]
                imgs_cams_list.append(img_cam)
            imgs_cams_tensor = torch.stack([transform(img) for img in imgs_cams_list], dim=0).to(device)
            
            h, w  = stride*(imgs_cams_tensor.shape[2]//stride), stride*(imgs_cams_tensor.shape[3]//stride)
            with torch.no_grad():
                features_cams = model.forward_features(imgs_cams_tensor[...,:h,:w])['x_norm_patchtokens']
                features_cams = features_cams.reshape(len(sensor_names),h//stride,w//stride,-1).cpu()
                
                
            # Get transformations.
            T_global_mainvehicle = scenes[scene_idx]['sample_lidar_T_global_vehicle'][sample_idx]
            T_mainvehicle_global = torch.linalg.inv(T_global_mainvehicle.clone())
            T_vehicle_lidar      = scenes[scene_idx]['T_vehicle_lidar']
            T_lidar_vehicle      = torch.linalg.inv(T_vehicle_lidar.clone())
            
            T_cam1_vehicle, T_camvehicle1_global, cam1_proj_matrix = get_camera_transformations(nusc=nusc, sample_record=sample_record, sensor_name=sensor_names[0])
            T_cam2_vehicle, T_camvehicle2_global, cam2_proj_matrix = get_camera_transformations(nusc=nusc, sample_record=sample_record, sensor_name=sensor_names[1])
            T_cam3_vehicle, T_camvehicle3_global, cam3_proj_matrix = get_camera_transformations(nusc=nusc, sample_record=sample_record, sensor_name=sensor_names[2])
            T_cam4_vehicle, T_camvehicle4_global, cam4_proj_matrix = get_camera_transformations(nusc=nusc, sample_record=sample_record, sensor_name=sensor_names[3])
            T_cam5_vehicle, T_camvehicle5_global, cam5_proj_matrix = get_camera_transformations(nusc=nusc, sample_record=sample_record, sensor_name=sensor_names[4])
            T_cam6_vehicle, T_camvehicle6_global, cam6_proj_matrix = get_camera_transformations(nusc=nusc, sample_record=sample_record, sensor_name=sensor_names[5])
            
            
            # Get LiDAR points.
            lidar_token = scenes[scene_idx]['sample_lidar_tokens'][sample_idx]
            pc_lidar    = get_lidar_sweep(nusc=nusc, lidar_token=lidar_token)
            pc_global   = (T_global_mainvehicle @ T_vehicle_lidar @ pc_lidar.T).T
            
            
            # Get cluster dict.
            lidar_record = nusc.get('sample_data', lidar_token)
            filename     = os.path.basename(lidar_record['filename']).replace('.pcd.bin','.npy')
            cluster_dict = np.load(os.path.join(intermediate_results_spatialclustering_dir, filename), allow_pickle=True).item()
            
            
            # Project points and get distance to image center.
            uvs_cam1, dist_to_cam1img_center = get_projections_and_distance_to_center(img_cam=imgs_cams_list[0], stride=stride, pc_global=pc_global.clone(), T_cam_vehicle=T_cam1_vehicle, T_camvehicle_global=T_camvehicle1_global, cam_proj_matrix=cam1_proj_matrix)
            uvs_cam2, dist_to_cam2img_center = get_projections_and_distance_to_center(img_cam=imgs_cams_list[1], stride=stride, pc_global=pc_global.clone(), T_cam_vehicle=T_cam2_vehicle, T_camvehicle_global=T_camvehicle2_global, cam_proj_matrix=cam2_proj_matrix)
            uvs_cam3, dist_to_cam3img_center = get_projections_and_distance_to_center(img_cam=imgs_cams_list[2], stride=stride, pc_global=pc_global.clone(), T_cam_vehicle=T_cam3_vehicle, T_camvehicle_global=T_camvehicle3_global, cam_proj_matrix=cam3_proj_matrix)
            uvs_cam4, dist_to_cam4img_center = get_projections_and_distance_to_center(img_cam=imgs_cams_list[3], stride=stride, pc_global=pc_global.clone(), T_cam_vehicle=T_cam4_vehicle, T_camvehicle_global=T_camvehicle4_global, cam_proj_matrix=cam4_proj_matrix)
            uvs_cam5, dist_to_cam5img_center = get_projections_and_distance_to_center(img_cam=imgs_cams_list[4], stride=stride, pc_global=pc_global.clone(), T_cam_vehicle=T_cam5_vehicle, T_camvehicle_global=T_camvehicle5_global, cam_proj_matrix=cam5_proj_matrix)
            uvs_cam6, dist_to_cam6img_center = get_projections_and_distance_to_center(img_cam=imgs_cams_list[5], stride=stride, pc_global=pc_global.clone(), T_cam_vehicle=T_cam6_vehicle, T_camvehicle_global=T_camvehicle6_global, cam_proj_matrix=cam6_proj_matrix)
            
            
            # Get closest image index for each projected point.
            dist_to_camimgs_center = torch.stack((dist_to_cam1img_center, dist_to_cam2img_center, dist_to_cam3img_center, dist_to_cam4img_center, dist_to_cam5img_center, dist_to_cam6img_center))
            mindist_cam_ids        = torch.argmin(dist_to_camimgs_center, dim=0)
            
            
            # Stack uvs arrays.
            uvs_cams = torch.stack((uvs_cam1, uvs_cam2, uvs_cam3, uvs_cam4, uvs_cam5, uvs_cam6))
            
            
            # Get features for all points (bilinear interpolation).
            features_cams = torch.nn.functional.pad(features_cams.permute(0,3,1,2), pad=(1,1,1,1), mode='replicate').permute(0,2,3,1)
            num_cams, map_height, map_width, feature_dim = features_cams.shape
            
            points_features = torch.zeros([uvs_cams.shape[1],feature_dim])
            project_ids     = torch.where(~torch.isposinf(torch.min(dist_to_camimgs_center, dim=0)[0]))[0]
            cam_ids         = mindist_cam_ids[project_ids]
            
            project_uvs     = uvs_cams[cam_ids,project_ids,:]/stride+1   # Add 1 to each index because we did padding with 1.
            
            project_uvs1    = torch.floor(torch.stack((project_uvs[:,0]-0.5+0, project_uvs[:,1]-0.5+0)).T).int()
            project_uvs2    = torch.floor(torch.stack((torch.clip(project_uvs[:,0]-0.5+1, min=0, max=map_width-1), project_uvs[:,1]-0.5+0)).T).int()
            project_uvs3    = torch.floor(torch.stack((torch.clip(project_uvs[:,0]-0.5+1, min=0, max=map_width-1), torch.clip(project_uvs[:,1]-0.5+1, min=0, max=map_height-1))).T).int()
            project_uvs4    = torch.floor(torch.stack((project_uvs[:,0]-0.5+0, torch.clip(project_uvs[:,1]-0.5+1, min=0, max=map_height-1))).T).int()
            project_res     = project_uvs-torch.floor(project_uvs)
            
            weight1 = (1-project_res[:,0])*(1-project_res[:,1])
            weight2 = project_res[:,0]*(1-project_res[:,1])
            weight3 = project_res[:,0]*project_res[:,1]
            weight4 = (1-project_res[:,0])*project_res[:,1]
            
            points_features[project_ids] = weight1.reshape(-1,1)*features_cams[cam_ids,project_uvs1[:,1],project_uvs1[:,0],:] + weight2.reshape(-1,1)*features_cams[cam_ids,project_uvs2[:,1],project_uvs2[:,0],:] + weight3.reshape(-1,1)*features_cams[cam_ids,project_uvs3[:,1],project_uvs3[:,0],:] + weight4.reshape(-1,1)*features_cams[cam_ids,project_uvs4[:,1],project_uvs4[:,0],:]
            
            
            # Get feature for each cluster.
            feature_dict = {}
            for cluster_idx in cluster_dict.keys():
                cluster = cluster_dict[cluster_idx]
                
                cluster_feature = points_features[cluster.idsall_frame[0]].mean(dim=0).numpy() if len(cluster.idsall_frame)>0 else None
                
                feature_info            = types.SimpleNamespace()
                feature_info.feature    = cluster_feature
                feature_info.valid      = not (cluster_feature==0).all() if cluster_feature is not None else False
                feature_info.num_points = len(cluster.idsall_frame[0])
                
                feature_dict[cluster_idx] = feature_info
                
                
            # Save dictionary.
            lidar_token  = scenes[scene_idx]['sample_lidar_tokens'][sample_idx]
            lidar_record = nusc.get('sample_data', lidar_token)
            filename     = os.path.basename(lidar_record['filename']).replace('.pcd.bin','.npy')
            np.save(os.path.join(intermediate_results_appearanceembedding_dir, filename), feature_dict)
