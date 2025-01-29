import numpy as np
import open3d
import os
import scipy
import torch
import types
from components.component_spatialclustering import fit_bounding_box
from nuscenes.nuscenes import NuScenes
from typing import Dict, List, Tuple
from utils.utils_functions import get_lidar_sweep



def toOpen3d(points):
    cloud        = open3d.geometry.PointCloud()
    cloud.points = open3d.utility.Vector3dVector(points[:,:3])
    return cloud



def get_histogram_based_and_icp_based_transformations(source_points:torch.Tensor, target_points:torch.Tensor, search_size:float, search_step:float, max_icp_iterations:int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute histogram-based and ICP-based homogeneous transformation matrices.
    
    Args:
        source_points (torch.Tensor) : Source LiDAR point cloud with shape (N,4).
        target_points (torch.Tensor) : Target LiDAR point cloud with shape (N,4).
        search_size (float) : Search size in x and y direction in meters.
        search_step (float) : Step size in x and y direction in meters.
        max_icp_iterations (int) : Maximum number of iterations for ICP.
        
    Returns:
        T_hist (torch.Tensor) : Histogram-based homogeneous transformation matrix.
        T_icp_3dof (torch.Tensor) : ICP-based homogeneous transformation matrix.
    """
    
    
    def unravel_index(index, shape):   # Torch 1.x has no torch.unravel_index.
        out = []
        for dim in reversed(shape):
            out.append(index % dim)
            index = index // dim
        return tuple(reversed(out))
    
    
    x_values = torch.linspace(-search_size, search_size, int(2*search_size/search_step+1))
    y_values = torch.linspace(-search_size, search_size, int(2*search_size/search_step+1))
    z_values = torch.linspace(-0.10, 0.10, 11)
    
    translation_vectors = target_points[:,:3].reshape(1,-1,3)-source_points[:,:3].reshape(-1,1,3)
    
    
    H, _ = torch.histogramdd(translation_vectors.reshape(-1,3), bins=(x_values, y_values, z_values))
    x_idx, y_idx, z_idx = unravel_index(torch.argmax(H).item(), H.shape)
    
    
    best_x = (x_values[x_idx]+x_values[x_idx+1])/2
    best_y = (y_values[y_idx]+y_values[y_idx+1])/2
    best_z = (z_values[z_idx]+z_values[z_idx+1])/2
    
    
    T_hist = torch.eye(4)
    T_hist[0,3] = best_x
    T_hist[1,3] = best_y
    T_hist[2,3] = best_z
    
    
    reg = open3d.pipelines.registration.registration_icp(source=toOpen3d(source_points.clone()),
                                                         target=toOpen3d(target_points.clone()),
                                                         max_correspondence_distance=2*search_step,
                                                         init=T_hist.clone(),
                                                         estimation_method=open3d.pipelines.registration.TransformationEstimationPointToPoint(),
                                                         criteria=open3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_icp_iterations))
    T_icp = torch.from_numpy(reg.transformation.copy())   # T_t1_t2 := t2 pose in t1 coordinate system.
    
    
    x_icp, y_icp, z_icp = T_icp[:3,3]
    yaw_radians = torch.arctan2(T_icp[1,0], T_icp[0,0])
    
    
    T_icp3dof = torch.eye(4)
    T_icp3dof[0,3]   = x_icp
    T_icp3dof[1,3]   = y_icp
    T_icp3dof[2,3]   = z_icp
    T_icp3dof[:2,:2] = torch.tensor([[torch.cos(yaw_radians), -torch.sin(yaw_radians)], [torch.sin(yaw_radians), torch.cos(yaw_radians)]])
    
    
    return T_hist, T_icp3dof



# ICP-based scene flow.
# Original version from https://github.com/yanconglin/ICP-Flow.
def scene_flow(pc_lidar:torch.Tensor, cluster:types.SimpleNamespace, Ts_coneplane_lidar:torch.Tensor, hyperparameters:Dict, timestamps:Dict) -> types.SimpleNamespace:
    """
    Estimate BEV velocity and fit new bounding box (for dynamic clusters).
    
    Args:
        pc_lidar (torch.Tensor) : LiDAR point cloud expressed in global frame with shape (N,4).
        cluster (types.SimpleNamespace) : Spatial cluster respresenting an object proposal.
        Ts_coneplane_lidar (torch.Tensor) : Tensor containing homogeneous transformation matrices for mapping 3D points from reference frame to plane frame with shape (num_cones,4,4).
        hyperparameters (dict) : Hyperparameters for scene flow.
        timestamps (dict) : Timestamp for relative frame indices.
        
    Returns:
        cluster (types.SimpleNamespace) : Spatial cluster respresenting an object proposal.
    """
    
    
    # Scene Flow: Step 0 (select frames and get indices).
    M = hyperparameters['Step0__M']   # Unit: 1.
    T = hyperparameters['Step0__T']   # Unit: 1.
    
    available_frames = torch.sort(torch.tensor(list(cluster.idsall_frame.keys()))).values.tolist()
    
    t0_frame   = 0
    t1_1_frame = available_frames[M-T]
    t1_2_frame = available_frames[M-T+1]
    t2_1_frame = available_frames[M+T]
    t2_2_frame = available_frames[M+T+1]
    
    t0_ids   = cluster.idsall_frame2[0]
    t1_1_ids = cluster.idsall_frame2[t1_1_frame]
    t1_2_ids = cluster.idsall_frame2[t1_2_frame]
    t2_1_ids = cluster.idsall_frame2[t2_1_frame]
    t2_2_ids = cluster.idsall_frame2[t2_2_frame]
    
    
    # Scene Flow: Step 1 (match first two and last two sweeps to get a denser sweep if individual sweeps are sparse).
    bottom_drop_thres       = hyperparameters['Step1__bottom_drop_thres']   # Unit: meters.
    top_drop_thres          = hyperparameters['Step1__top_drop_thres']   # Unit: meters.
    min_points_per_pc_thres = hyperparameters['Step1__min_points_per_pc_thres']   # Unit: 1.
    search_size             = hyperparameters['Step1__search_size']   # Unit: meters.
    search_step             = hyperparameters['Step1__search_step']   # Unit: meters.
    max_icp_iterations      = hyperparameters['Step1__max_icp_iterations']   # Unit: 1.
    max_dist_inlier_thres   = hyperparameters['Step1__max_dist_inlier_thres']   # Unit: meters.
    max_pc_size             = hyperparameters['Step1__max_pc_size']   # Unit: 1.
    
    T_bbox_lidar = torch.linalg.inv(torch.from_numpy(cluster.touchground_T_lidar_bbox))
    matched1, matched2 = False, False
    
    if len(t1_1_ids)>0:
        source_points_bbox1 = (T_bbox_lidar @ pc_lidar[t1_1_ids].clone().T).T
        source_points_bbox1 = source_points_bbox1[(source_points_bbox1[:,2]>bottom_drop_thres) & (source_points_bbox1[:,2]<top_drop_thres),:]
        
        if source_points_bbox1.shape[0]>=min_points_per_pc_thres:
            T_icp1       = torch.eye(4)   # T_source_target := target pose in source coordinate system.
            points_bbox1 = source_points_bbox1.clone()
            matched1     = True
            
        elif len(t1_2_ids)>0:
            target_points_bbox1 = (T_bbox_lidar @ pc_lidar[t1_2_ids].clone().T).T
            target_points_bbox1 = target_points_bbox1[(target_points_bbox1[:,2]>bottom_drop_thres) & (target_points_bbox1[:,2]<top_drop_thres),:]
            
            if source_points_bbox1.shape[0]>=min_points_per_pc_thres/2 and target_points_bbox1.shape[0]>=min_points_per_pc_thres/2:
                T_hist, T_icp3dof = get_histogram_based_and_icp_based_transformations(source_points=source_points_bbox1.clone(), target_points=target_points_bbox1.clone(), search_size=search_size, search_step=search_step, max_icp_iterations=max_icp_iterations,)
                
                dists                  = torch.from_numpy(scipy.spatial.distance.cdist(source_points_bbox1.clone()[:,:3], (torch.linalg.inv(T_hist) @ target_points_bbox1.clone().T).T[:,:3], metric='euclidean')).float()
                min_dists1, min_dists2 = torch.min(dists, dim=1).values, torch.min(dists, dim=0).values
                inlier_cnt__hist       = torch.mean(torch.tensor([(min_dists1<=max_dist_inlier_thres).sum(), (min_dists2<=max_dist_inlier_thres).sum()]).float()).item()
                
                dists                  = torch.from_numpy(scipy.spatial.distance.cdist(source_points_bbox1.clone()[:,:3], (torch.linalg.inv(T_icp3dof) @ target_points_bbox1.clone().T).T[:,:3], metric='euclidean')).float()
                min_dists1, min_dists2 = torch.min(dists, dim=1).values, torch.min(dists, dim=0).values
                inlier_cnt__icp3dof    = torch.mean(torch.tensor([(min_dists1<=max_dist_inlier_thres).sum(), (min_dists2<=max_dist_inlier_thres).sum()]).float()).item()
                
                if max(inlier_cnt__hist, inlier_cnt__icp3dof)>=min_points_per_pc_thres/2:
                    T_icp1       = T_icp3dof if inlier_cnt__icp3dof>=inlier_cnt__hist else T_hist   # T_source_target := target pose in source coordinate system.
                    points_bbox1 = torch.concatenate((source_points_bbox1, (torch.linalg.inv(T_icp1) @ target_points_bbox1.T).T), dim=0)
                    matched1     = True
                    
    if len(t2_1_ids)>0:
        source_points_bbox2 = (T_bbox_lidar @ pc_lidar[t2_1_ids].clone().T).T
        source_points_bbox2 = source_points_bbox2[(source_points_bbox2[:,2]>bottom_drop_thres) & (source_points_bbox2[:,2]<top_drop_thres),:]
        
        if source_points_bbox2.shape[0]>=min_points_per_pc_thres:
            T_icp2       = torch.eye(4)   # T_source_target := target pose in source coordinate system.
            points_bbox2 = source_points_bbox2.clone()
            matched2     = True
            
        elif len(t2_2_ids)>0:
            target_points_bbox2 = (T_bbox_lidar @ pc_lidar[t2_2_ids].clone().T).T
            target_points_bbox2 = target_points_bbox2[(target_points_bbox2[:,2]>bottom_drop_thres) & (target_points_bbox2[:,2]<top_drop_thres),:]
            
            if source_points_bbox2.shape[0]>=min_points_per_pc_thres/2 and target_points_bbox2.shape[0]>=min_points_per_pc_thres/2:
                T_hist, T_icp3dof = get_histogram_based_and_icp_based_transformations(source_points=source_points_bbox2.clone(), target_points=target_points_bbox2.clone(), search_size=search_size, search_step=search_step, max_icp_iterations=max_icp_iterations,)
                
                dists                  = torch.from_numpy(scipy.spatial.distance.cdist(source_points_bbox2.clone()[:,:3], (torch.linalg.inv(T_hist) @ target_points_bbox2.clone().T).T[:,:3], metric='euclidean')).float()
                min_dists1, min_dists2 = torch.min(dists, dim=1).values, torch.min(dists, dim=0).values
                inlier_cnt__hist       = torch.mean(torch.tensor([(min_dists1<=max_dist_inlier_thres).sum(), (min_dists2<=max_dist_inlier_thres).sum()]).float()).item()
                
                dists                  = torch.from_numpy(scipy.spatial.distance.cdist(source_points_bbox2.clone()[:,:3], (torch.linalg.inv(T_icp3dof) @ target_points_bbox2.clone().T).T[:,:3], metric='euclidean')).float()
                min_dists1, min_dists2 = torch.min(dists, dim=1).values, torch.min(dists, dim=0).values
                inlier_cnt__icp3dof    = torch.mean(torch.tensor([(min_dists1<=max_dist_inlier_thres).sum(), (min_dists2<=max_dist_inlier_thres).sum()]).float()).item()
                
                if max(inlier_cnt__hist, inlier_cnt__icp3dof)>=min_points_per_pc_thres/2:
                    T_icp2       = T_icp3dof if inlier_cnt__icp3dof>=inlier_cnt__hist else T_hist   # T_source_target := target pose in source coordinate system.
                    points_bbox2 = torch.concatenate((source_points_bbox2, (torch.linalg.inv(T_icp2) @ target_points_bbox2.T).T), axis=0)
                    matched2     = True
                    
    if matched1 and matched2:
        torch.manual_seed(0)
        source_points_bbox3 = points_bbox1.clone() if points_bbox1.shape[0]<=max_pc_size else points_bbox1.clone()[((1/points_bbox1.shape[0])*torch.ones([points_bbox1.shape[0]])).multinomial(num_samples=max_pc_size, replacement=False)]
        target_points_bbox3 = points_bbox2.clone() if points_bbox2.shape[0]<=max_pc_size else points_bbox2.clone()[((1/points_bbox2.shape[0])*torch.ones([points_bbox2.shape[0]])).multinomial(num_samples=max_pc_size, replacement=False)]
        
        
    # Scene FLow: Step 2 (Assume static object: match dense source and target).
    max_dist_inlier_thres = hyperparameters['Step2__max_dist_inlier_thres']   # Unit: meters.
    
    static__success = False
    static__inliers = 0
    
    if matched1 and matched2:
        dists                  = torch.from_numpy(scipy.spatial.distance.cdist(source_points_bbox3.clone()[:,:3], target_points_bbox3.clone()[:,:3], metric='euclidean'))
        min_dists1, min_dists2 = torch.min(dists, dim=1).values, torch.min(dists, dim=0).values
        inlier_cnt             = torch.mean(torch.tensor([(min_dists1<=max_dist_inlier_thres).sum(), (min_dists2<=max_dist_inlier_thres).sum()]).float()).item()
        
        if inlier_cnt>0:
            static__inliers = inlier_cnt
            static__T_icp   = torch.eye(4)   # T_t1_t2 := t2 pose in t1 coordinate system.
            static__success = True
            
            
    # Scene Flow: Step 3 (Assume dynamic object: get good initialization and match dense source and target).
    search_size           = hyperparameters['Step3__search_size']   # Unit: meters.
    search_step           = hyperparameters['Step3__search_step']   # Unit: meters.
    max_icp_iterations    = hyperparameters['Step3__max_icp_iterations']   # Unit: 1.
    max_dist_inlier_thres = hyperparameters['Step3__max_dist_inlier_thres']   # Unit: meters.
    
    dynamic__success = False
    dynamic__inliers = 0
    
    if matched1 and matched2:
        T_hist, T_icp3dof = get_histogram_based_and_icp_based_transformations(source_points=source_points_bbox3.clone(), target_points=target_points_bbox3.clone(), search_size=search_size, search_step=search_step, max_icp_iterations=max_icp_iterations,)
        
        dists                  = torch.from_numpy(scipy.spatial.distance.cdist(source_points_bbox3.clone()[:,:3], (torch.linalg.inv(T_hist) @ target_points_bbox3.clone().T).T[:,:3], metric='euclidean')).float()
        min_dists1, min_dists2 = torch.min(dists, dim=1).values, torch.min(dists, dim=0).values
        inlier_cnt__hist       = torch.mean(torch.tensor([(min_dists1<=max_dist_inlier_thres).sum(), (min_dists2<=max_dist_inlier_thres).sum()]).float()).item()
        
        dists                  = torch.from_numpy(scipy.spatial.distance.cdist(source_points_bbox3.clone()[:,:3], (torch.linalg.inv(T_icp3dof) @ target_points_bbox3.clone().T).T[:,:3], metric='euclidean')).float()
        min_dists1, min_dists2 = torch.min(dists, dim=1).values, torch.min(dists, dim=0).values
        inlier_cnt__icp3dof    = torch.mean(torch.tensor([(min_dists1<=max_dist_inlier_thres).sum(), (min_dists2<=max_dist_inlier_thres).sum()]).float()).item()
        
        inlier_cnt = torch.max(torch.tensor([inlier_cnt__hist, inlier_cnt__icp3dof]).float()).item()
        if inlier_cnt>0:
            dynamic__inliers = inlier_cnt
            dynamic__T_icp   = T_icp3dof if inlier_cnt__icp3dof>=inlier_cnt__hist else T_hist   # T_t1_t2 := t2 pose in t1 coordinate system.
            dynamic__success = True
            
            
    # Scene Flow: Step 4 (compute velocity and fit new bounding box for dynamic cluster).
    num_cones  = hyperparameters['Step4__num_cones']   # Unit: 1.
    lidar_freq = hyperparameters['Step4__lidar_frequency']   # Unit: 1.
    
    fov_cone = 360/num_cones   # Unit: degrees.
    
    if dynamic__success and dynamic__inliers>=static__inliers:
        T_equivalent = (torch.round(torch.tensor(timestamps[t2_1_frame]-timestamps[t1_1_frame])/(1/lidar_freq))/2).int().item()
        
        T_lidar_bbox0  = torch.from_numpy(cluster.touchground_T_lidar_bbox)
        onestep__T_icp = torch.from_numpy(scipy.linalg.expm(scipy.linalg.logm(dynamic__T_icp, disp=False)[0]/(2*T_equivalent))).real.float()   # So "dynamic__T_icp = np.linalg.matrix_power(onestep__T_icp, 2*T)".
        
        t1_relative_steps = torch.round(torch.tensor(timestamps[t1_1_frame]-timestamps[t0_frame])/(1/lidar_freq)).int().item()
        t2_relative_steps = torch.round(torch.tensor(timestamps[t2_1_frame]-timestamps[t0_frame])/(1/lidar_freq)).int().item()
        
        pc_bbox0       = (torch.linalg.inv(T_lidar_bbox0) @ pc_lidar.clone().T).T
        t1_1__pc_bbox0 = (torch.linalg.matrix_power(onestep__T_icp, -t1_relative_steps) @ pc_bbox0[t1_1_ids].T).T
        t1_2__pc_bbox0 = (torch.linalg.matrix_power(onestep__T_icp, -t1_relative_steps) @ torch.linalg.inv(T_icp1) @ pc_bbox0[t1_2_ids].T).T if len(t1_2_ids)>0 else torch.zeros([0,4])
        t2_1__pc_bbox0 = (torch.linalg.matrix_power(onestep__T_icp, -t2_relative_steps) @ pc_bbox0[t2_1_ids].T).T
        t2_2__pc_bbox0 = (torch.linalg.matrix_power(onestep__T_icp, -t2_relative_steps) @ torch.linalg.inv(T_icp2) @ pc_bbox0[t2_2_ids].T).T if len(t2_2_ids)>0 else torch.zeros([0,4])
        
        cluster_pc_bbox0 = torch.concatenate((t1_1__pc_bbox0, t1_2__pc_bbox0, t2_1__pc_bbox0, t2_2__pc_bbox0), dim=0)
        cluster_pc_lidar = (T_lidar_bbox0 @ cluster_pc_bbox0.T).T
        
        sceneflow__T_lidar_bbox, sceneflow__bboxdimensions = fit_bounding_box(cluster_pc_lidar.clone())
        
        cone_idx            = ((((180/np.pi*torch.arctan2(sceneflow__T_lidar_bbox[1,3], sceneflow__T_lidar_bbox[0,3])+360)%360)/fov_cone)%num_cones).int().item()
        T_coneplane_lidar   = Ts_coneplane_lidar[cone_idx]
        height_above_ground = (T_coneplane_lidar @ sceneflow__T_lidar_bbox)[2,3].item()
        
        sceneflow__touchground_T_lidar_bbox       = sceneflow__T_lidar_bbox.clone()
        sceneflow__touchground_T_lidar_bbox[2,3] += -height_above_ground
        sceneflow__touchground_bboxdimensions     = sceneflow__bboxdimensions.copy()
        sceneflow__touchground_bboxdimensions[2] += height_above_ground
        
        R_lidar_bbox0        = torch.eye(4)
        R_lidar_bbox0[:3,:3] = T_lidar_bbox0[:3,:3]
        
        moving_direction_lidar = (R_lidar_bbox0 @ dynamic__T_icp)[:2,3]
        velocity_lidar         = (moving_direction_lidar/(timestamps[t2_1_frame]-timestamps[t1_1_frame])).tolist()
        velocity_magnitude     = torch.linalg.norm(torch.tensor(velocity_lidar), ord=2).item()
        
        sceneflow_initial__touchground_yaw_radians = torch.arctan2(sceneflow__touchground_T_lidar_bbox[1,0], sceneflow__touchground_T_lidar_bbox[0,0]).item()
        moving_direction_based_yaw_radians         = torch.arctan2(moving_direction_lidar[1], moving_direction_lidar[0]).item()
        yaw_options                                = torch.tensor([sceneflow_initial__touchground_yaw_radians+i*np.pi/2 for i in range(4)])
        sceneflow__touchground_yaw_radians         = yaw_options[torch.argmin(torch.min(torch.stack(((yaw_options-moving_direction_based_yaw_radians)%(2*np.pi), (moving_direction_based_yaw_radians-yaw_options)%(2*np.pi))), dim=0).values).item()].item()
        sceneflow__touchground_yaw_radians         = sceneflow__touchground_yaw_radians-2*np.pi if sceneflow__touchground_yaw_radians>np.pi else sceneflow__touchground_yaw_radians
        
        c, s = torch.cos(torch.tensor(sceneflow__touchground_yaw_radians)), torch.sin(torch.tensor(sceneflow__touchground_yaw_radians))
        R_lidar_bbox = torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1],])
        sceneflow__touchground_T_lidar_bbox[:3,:3] = R_lidar_bbox
        
        angle_difference = min((sceneflow__touchground_yaw_radians-sceneflow_initial__touchground_yaw_radians)%(2*np.pi), (sceneflow_initial__touchground_yaw_radians-sceneflow__touchground_yaw_radians)%(2*np.pi))
        if torch.isclose(torch.tensor(angle_difference), torch.tensor(np.pi/2)):
            L, W = sceneflow__touchground_bboxdimensions[:2]
            sceneflow__touchground_bboxdimensions[0] = W
            sceneflow__touchground_bboxdimensions[1] = L
            
        sceneflow__touchground_T_lidar_bbox = sceneflow__touchground_T_lidar_bbox.numpy()
        
    else:
        sceneflow__touchground_T_lidar_bbox   = cluster.touchground_T_lidar_bbox
        sceneflow__touchground_bboxdimensions = cluster.touchground_bboxdimensions
        sceneflow__touchground_yaw_radians    = cluster.touchground_yaw_radians
        height_above_ground                   = cluster.height_above_ground
        velocity_lidar                        = [0,0]
        velocity_magnitude                    = 0
        
    cluster.sceneflow__touchground_T_lidar_bbox   = sceneflow__touchground_T_lidar_bbox
    cluster.sceneflow__touchground_bboxdimensions = sceneflow__touchground_bboxdimensions
    cluster.sceneflow__touchground_yaw_radians    = sceneflow__touchground_yaw_radians
    cluster.height_above_ground                   = height_above_ground
    cluster.velocity_lidar                        = velocity_lidar
    cluster.velocity_magnitude                    = velocity_magnitude
    cluster.num_inliers                           = max(static__inliers, dynamic__inliers)
    
    
    return cluster



def main__scene_flow(nusc:NuScenes, scenes:List, hyperparameters:Dict, intermediate_results_groundremoval_dir2:str, intermediate_results_spatialclustering_dir:str, intermediate_results_sceneflow_dir:str, first_scene:int=0, num_of_scenes:int=850):
    """
    Compute velocity for all clusters.
    
    Args:
        nusc (NuScenes) : NuScenes Object.
        scenes (list) : List with scene dictionaries of dataset.
        hyperparameters (dict) : Hyperparameters for scene flow.
        intermediate_results_groundremoval_dir2 (str) : Folder for ground removal results (Ts_coneplane_lidar).
        intermediate_results_spatialclustering_dir (str) : Folder for spatial clustering results (cluster dicts).
        intermediate_results_sceneflow_dir (str) : Folder for scene flow results (cluster dicts).
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
            pc_lidar = get_lidar_sweep(nusc=nusc, lidar_token=lidar_token)
            
            
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
                
                
            # Get timestamps.
            M = hyperparameters['Step0__M']
            T = hyperparameters['Step0__T']
            
            available_frames = extra_ids
            available_frames.append(sweep_idx)
            available_frames = torch.sort(torch.IntTensor(available_frames)).values.tolist()
            
            relative_ids = (torch.IntTensor(available_frames)-sweep_idx).tolist()
            
            timestamps = {relative_ids[available_frames.index(sweep_idx)]: scenes[scene_idx]['sweep_lidar_timestamps'][sweep_idx]/(1e6),
                          relative_ids[available_frames.index(available_frames[M-T])]: scenes[scene_idx]['sweep_lidar_timestamps'][available_frames[M-T]]/(1e6),
                          relative_ids[available_frames.index(available_frames[M+T])]: scenes[scene_idx]['sweep_lidar_timestamps'][available_frames[M+T]]/(1e6),}
            
            
            # Get ground plane.
            lidar_token        = scenes[scene_idx]['sample_lidar_tokens'][sample_idx]
            lidar_record       = nusc.get('sample_data', lidar_token)
            filename           = os.path.basename(lidar_record['filename']).replace('.pcd.bin','.npy')
            Ts_coneplane_lidar = torch.from_numpy(np.load(os.path.join(intermediate_results_groundremoval_dir2, filename)))
            
            
            # Get cluster dict.
            lidar_token  = scenes[scene_idx]['sample_lidar_tokens'][sample_idx]
            lidar_record = nusc.get('sample_data', lidar_token)
            filename     = os.path.basename(lidar_record['filename']).replace('.pcd.bin','.npy')
            cluster_dict = np.load(os.path.join(intermediate_results_spatialclustering_dir, filename), allow_pickle=True).item()
            
            
            # Compute velocity for each cluster.
            for cluster_idx in list(cluster_dict.keys()):
                cluster = scene_flow(pc_lidar=pc_lidar, cluster=cluster_dict[cluster_idx], Ts_coneplane_lidar=Ts_coneplane_lidar, hyperparameters=hyperparameters, timestamps=timestamps)
                cluster_dict[cluster_idx] = cluster
                
                
            # Save dictionary (everything).
            lidar_token  = scenes[scene_idx]['sample_lidar_tokens'][sample_idx]
            lidar_record = nusc.get('sample_data', lidar_token)
            filename     = os.path.basename(lidar_record['filename']).replace('.pcd.bin','.npy')
            np.save(os.path.join(intermediate_results_sceneflow_dir, filename), cluster_dict)
            
            
            # Save disctionary (drop point indices; for fast loading).
            for cluster_idx in cluster_dict.keys():
                cluster_dict[cluster_idx].idsall_aggregated = None
                cluster_dict[cluster_idx].idsall_frame = None
                cluster_dict[cluster_idx].idsall_aggregated2 = None
                cluster_dict[cluster_idx].idsall_frame2 = None
            lidar_token  = scenes[scene_idx]['sample_lidar_tokens'][sample_idx]
            lidar_record = nusc.get('sample_data', lidar_token)
            filename     = os.path.basename(lidar_record['filename']).replace('.pcd.bin','__small.npy')
            np.save(os.path.join(intermediate_results_sceneflow_dir, filename), cluster_dict)
