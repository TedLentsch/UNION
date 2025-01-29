import numpy as np
import os
import time
import torch
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import train, val, test, mini_train, mini_val
from typing import Dict, List



class KMeans_Torch():
    """
    K-Means clustering using PyTorch.
    """
    
    
    def __init__(self, init:str='k-means++', num_clusters:int=20, max_iterations:int=5000, num_init:int=10, device:None=torch.device('cpu')):
        """
        Initialize K-Means clustering instance.
        
        Args:
            init (str) : Initialization strategy. There are two options: (1) random and (2) sampling based on an empirical probability distribution.
            num_clusters (int) : Number of clusters for clustering.
            max_iterations (int) : Maximum number of interations for updating cluster centers.
            num_init (int) : Number of times to initialize clustering.
            device (torch.device) : Device for processing data.
        """
        
        
        assert init in ['random', 'k-means++'], print('Wrong init setting given! Provide ``random`` or ``k-means++``!')
        assert type(num_clusters)==int and num_clusters>=2, print('Number of clusters should be integer and at least equal to 2!')
        assert type(max_iterations)==int and max_iterations>=1, print('Maximum number of interations should be integer and at least equal to 1!')
        assert type(num_init)==int and num_init>=1, print('Number of times to initialize clustering should be integer and at least equal to 1!')
        
        
        self.init           = init
        self.num_clusters   = num_clusters
        self.max_iterations = max_iterations
        self.num_init       = num_init
        self.device         = device
        
        
        self.feature_dim = None
        self.centers     = None
        self.fitted      = None
        
        
    def __init_kmeans_random(self,  X:torch.Tensor, set_seed:bool=True, random_seed:int=0) -> torch.Tensor:
        """
        Initialize cluster centers using randomness.
        
        Args:
            X (torch.Tensor) : Input tensor with shape (N,feature_dim).
            set_seed (bool) : Set random seed for first fit. There are ``self.num_init`` fits.
            random_seed (int) : Random seed.
            
        Returns:
            centers (torch.Tensor) : Tensor containing initial cluster centers with shape (num_clusters,feature_dim).
        """
        
        
        # Random seed.
        if set_seed:
            torch.manual_seed(random_seed)
            torch.cuda.manual_seed(random_seed)
            torch.cuda.manual_seed_all(random_seed)
            
            
        # Get shape.
        num_samples, self.feature_dim = X.shape
        
        
        # Compute centers.
        centers = X[torch.randperm(num_samples)[:self.num_clusters]].clone()
        
        
        return centers
    
    
    def __init_kmeans_plusplus(self,  X:torch.Tensor, set_seed:bool=True, random_seed:int=0) -> torch.Tensor:
        """
        Initialize cluster centers using KMeans++.
        
        Args:
            X (torch.Tensor) : Input tensor with shape (N,feature_dim).
            set_seed (bool) : Set random seed for first fit. There are ``self.num_init`` fits.
            random_seed (int) : Random seed.
            
        Returns:
            centers (torch.Tensor) : Tensor containing initial cluster centers with shape (num_clusters,feature_dim).
        """
        
        
        # Random seed.
        if set_seed:
            torch.manual_seed(random_seed)
            torch.cuda.manual_seed(random_seed)
            torch.cuda.manual_seed_all(random_seed)
            
            
        # Get shape.
        num_samples, self.feature_dim = X.shape
        
        
        # Choose first center randomly.
        centers    = torch.zeros([self.num_clusters, self.feature_dim], device=self.device)
        centers[0] = X[torch.randint(0, num_samples, [1])].clone()
        
        
        # Compute other centers.
        closest_dist_squared = torch.cdist(x1=X, x2=centers[0:1], p=2).squeeze().pow(2)
        
        for i in range(1, self.num_clusters):
            probs      = closest_dist_squared / torch.sum(closest_dist_squared)
            centers[i] = X[torch.multinomial(probs, 1)].clone()
            
            new_closest_dist_squared = torch.cdist(x1=X, x2=centers[i:i+1], p=2).squeeze().pow(2)
            closest_dist_squared     = torch.min(closest_dist_squared, new_closest_dist_squared)
            
            
        return centers
    
    
    def fit(self, X:torch.Tensor, set_seed:bool=True, random_seed:int=0):
        """
        Compute K-Means clustering for input tensor.
        
        Args:
            X (torch.Tensor) : Input tensor with shape (N,feature_dim).
            set_seed (bool) : Set random seed for first fit. There are ``self.num_init`` fits.
            random_seed (int) : Random seed.
        """
        
        
        assert type(X)==torch.Tensor, print('Provide tensor as input for clustering!')
        assert len(X.shape)==2, print('Provide 2D tensor as input for clustering!')
        assert X.shape[0]>=1 and X.shape[1]>=1, print('Provide 2D tensor with at least single sample and at least single feature!')
        assert X.shape[0]>=self.num_clusters, print('Number of clusters cannot be smaller than number of samples!')
        assert type(set_seed)==bool, print('Variable ``set_seed`` should be boolean!')
        
        
        # Prepare data.
        X = X.clone().to(self.device)
        
        
        # Initialize multiple times.
        print('K-Means: Start fitting!\n')
        best_centers = None
        best_inertia = float('inf')
        for idx in range(self.num_init):
            
            
            # Time it.
            start_time = time.time()
            
            
            # Get centers.
            if idx==0:
                centers = self.__init_kmeans_random(X=X, set_seed=set_seed, random_seed=random_seed) if self.init=='random' else self.__init_kmeans_plusplus(X=X, set_seed=set_seed, random_seed=random_seed)
            else:
                centers = self.__init_kmeans_random(X=X, set_seed=False) if self.init=='random' else self.__init_kmeans_plusplus(X=X, set_seed=False)
                
                
            # Update algorithm.
            for n in range(self.max_iterations):
                
                
                # Assign labels.
                center_dists = torch.cdist(X, centers, p=2)
                labels       = torch.argmin(center_dists, dim=1)
                
                
                # Update centers.
                new_centers = torch.stack([X[labels==i].mean(dim=0) if (labels==i).sum()>0 else centers[i] for i in range(self.num_clusters)])
                
                
                # Check convergence.
                if torch.allclose(centers, new_centers):
                    print(f'Convergence reached after {n+1} iterations!')
                    break
                    
                    
                # Overwrite centers.
                centers = new_centers.clone()
                
                
            # Compute inertia.
            center_dists = torch.cdist(x1=X, x2=centers, p=2)
            min_dists    = torch.min(center_dists, dim=1).values
            inertia      = torch.sum(torch.pow(min_dists, exponent=2)).item()
            
            
            # Overwrite best ones.
            if inertia<best_inertia:
                best_centers = centers.clone()
                best_inertia = inertia
                
                
            # Logging.
            print(f'Iteration {str(idx).zfill(3)}: inertia = {inertia} | time = {time.time()-start_time} sec\n')
            
            
        # Overwrite centers.
        self.centers = best_centers
        
        
        # Fitting done.
        self.fitted = True
        
        
    def predict(self, X:torch.Tensor) -> torch.Tensor:
        """
        Predict labels based on fitted centers.
        
        Args:
            X (torch.Tensor) : Input tensor with shape (N,feature_dim).
            
        Returns:
            labels (torch.Tensor) : Label tensor with shape (N).
        """
        
        
        assert self.fitted, print('Do fitting first!')
        assert type(X)==torch.Tensor, print('Provide tensor as input for clustering!')
        assert len(X.shape)==2, print('Provide 2D tensor as input for clustering!')
        assert X.shape[0]>=1, print('Provide 2D tensor with at least single!')
        assert X.shape[1]==self.feature_dim, print('Provide samples with correct number of features!')
        
        
        # Prepare data.
        X = X.clone().to(self.device)
        
        
        # Determine closest center for samples.
        center_dists = torch.cdist(x1=X, x2=self.centers, p=2)
        labels       = torch.argmin(center_dists, dim=1)
        
        
        return labels
    
    
    def fit_predict(self, X:torch.Tensor, set_seed:bool=True, random_seed:int=0) -> torch.Tensor:
        """
        Compute K-Means clustering for input tensor and predict labels based on fitted centers.
        
        Args:
            X (torch.Tensor) : Input tensor with shape (N,feature_dim).
            set_seed (bool) : Set random seed for first fit. There are ``self.num_init`` fits.
            random_seed (int) : Random seed.
            
        Returns:
            labels (torch.Tensor) : Label tensor with shape (N).
        """
        
        
        # Fit data.
        self.fit(X=X, set_seed=set_seed, random_seed=random_seed)
        
        
        # Get labels.
        labels = self.predict(X=X)
        
        
        return labels



def main__appearance_clustering(nusc:NuScenes, scenes:List, hyperparameters:Dict, intermediate_results_sceneflow_dir:str, intermediate_results_appearanceembedding_dir:str, intermediate_results_appearanceclustering_dir:str, first_scene:int=0, num_of_scenes:int=850):
    """
    Cluster appearance embedding to obtain mobile objects.
    
    Args:
        nusc (NuScenes) : NuScenes Object.
        scenes (list) : List with scene dictionaries of dataset.
        hyperparameters (dict) : Hyperparameters for appearance clustering.
        intermediate_results_sceneflow_dir (str) : Folder for scene flow results (cluster dicts).
        intermediate_results_appearanceembedding_dir (str) : Folder for appearance embedding results (cluster dicts).
        intermediate_results_appearanceclustering_dir (str) : Folder for appearance clustering results (cluster dicts).
        first_scene (int) : Index of first scene for removing ground points.
        num_of_scenes (int) : Number of scenes for removing ground points.
    """
    
    
    # Step 0: Load object proposal features and velocities.
    print('Load appearance embeddings and estimated velocities!\n')
    
    feature_dim = hyperparameters['Step0__feature_dim']
    
    nested_sample_tokens = [scenes[scene_idx]['sample_tokens'] for scene_idx in range(first_scene, num_of_scenes) if scenes[scene_idx]['scene_name'] in train]
    sample_tokens__train = [token for tokens in nested_sample_tokens for token in tokens]
    
    cluster_dicts, feature_dicts = [], []
    for sample_token in sample_tokens__train:
        lidar_token  = nusc.get('sample', sample_token)['data']['LIDAR_TOP']
        lidar_record = nusc.get('sample_data', lidar_token)
        filename1    = os.path.basename(lidar_record['filename']).replace('.pcd.bin','__small.npy')
        filename2    = os.path.basename(lidar_record['filename']).replace('.pcd.bin','.npy')
        
        cluster_dict = np.load(os.path.join(intermediate_results_sceneflow_dir, filename1), allow_pickle=True).item()
        feature_dict = np.load(os.path.join(intermediate_results_appearanceembedding_dir, filename2), allow_pickle=True).item()
        
        cluster_dicts.append(cluster_dict)
        feature_dicts.append(feature_dict)
        
    X_valids, X_features, X_velocities, X_inliers = [], [], [], []
    token_to_clusterids_dict, idx_to_token_dict, idx_to_clusteridx_dict = {}, {}, {}
    for token, cluster_dict, feature_dict in zip(sample_tokens__train, cluster_dicts, feature_dicts):
        valids     = [feature_dict[cluster_idx].valid for cluster_idx in feature_dict.keys()]
        features   = [torch.from_numpy(feature_dict[cluster_idx].feature) if feature_dict[cluster_idx].valid else torch.zeros([feature_dim]) for cluster_idx in feature_dict.keys()]
        velocities = [cluster_dict[cluster_idx].velocity_magnitude for cluster_idx in cluster_dict.keys()]
        inliers    = [cluster_dict[cluster_idx].num_inliers for cluster_idx in cluster_dict.keys()]
        
        token_to_clusterids_dict[token] = list(range(len(X_features), len(X_features)+len(features)))
        for cluster_idx, idx in enumerate(range(len(X_features), len(X_features)+len(features))):
            idx_to_token_dict[idx]      = token
            idx_to_clusteridx_dict[idx] = cluster_idx
            
        X_valids.extend(valids)
        X_features.extend(features)
        X_velocities.extend(velocities)
        X_inliers.extend(inliers)
        
    X_valids      = torch.tensor(X_valids)
    X_features    = torch.stack(X_features)
    X_velocities  = torch.tensor(X_velocities)
    X_inliers     = torch.tensor(X_inliers)
    
    
    # Step 1: Cluster appearance embeddings.
    print('Cluster appearance embeddings!\n')
    
    K__class_agnostic = hyperparameters['Step1__K__class_agnostic']   # Unit: 1.
    max_iterations    = hyperparameters['Step1__max_iterations']   # Unit: 1.
    num_init          = hyperparameters['Step1__num_init']   # Unit: 1.
    
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    
    cluster_instance = KMeans_Torch(init='k-means++', num_clusters=K__class_agnostic, max_iterations=max_iterations, num_init=num_init, device=device)
    cluster_instance.fit(X=X_features.clone()[X_valids], set_seed=True, random_seed=0)
    
    labels__class_agnostic = cluster_instance.predict(X=X_features.clone()).cpu()
    
    centers  = cluster_instance.centers.clone().cpu().numpy()
    filename = os.path.join(intermediate_results_appearanceclustering_dir, f'K-means-centers__K-class-agnostic{str(K__class_agnostic).zfill(3)}__class-agnostic.npy')
    np.save(filename, centers)
    
    
    # Step 2: Select mobile clusters and get instances.
    print('Select mobile appearance clusters!\n')
    
    velocity_thres        = hyperparameters['Step2__velocity_thres']   # Unit: m/s.
    moving_fraction_thres = hyperparameters['Step2__moving_fraction_thres']   # Unit: 1.
    
    velocityfraction_per_cluster, sizes_per_cluster = {}, {}
    for label in torch.unique(labels__class_agnostic).tolist():
        label_bool                          = (labels__class_agnostic==label) & X_valids
        velocityfraction_per_cluster[label] = (((X_velocities[label_bool]>=velocity_thres) & (X_valids[label_bool])).sum() / label_bool.sum()).item()
        sizes_per_cluster[label]            = label_bool.sum().item()
        
    selected_labels__class_agnostic  = [label for label in list(range(K__class_agnostic)) if velocityfraction_per_cluster[label]>=moving_fraction_thres]
    use_cluster_bool__class_agnostic = (torch.isin(labels__class_agnostic, torch.tensor(selected_labels__class_agnostic)) & X_valids).tolist()
    
    
    # Step 3: Save clustering results (class-agnostic).
    print('Save clustering result for class-agnostic!\n')
    
    for cnt, sample_token in enumerate(sample_tokens__train):
        if cnt%50==0:
            print(f'sample_token: {cnt+1}/{len(sample_tokens__train)}')
            
        usage_dict__class_agnostic, label_dict__class_agnostic = {}, {}
        for cluster_idx, usage_idx in enumerate(token_to_clusterids_dict[sample_token]):
            usage_dict__class_agnostic[cluster_idx] = use_cluster_bool__class_agnostic[usage_idx]
            label_dict__class_agnostic[cluster_idx] = (labels__class_agnostic[usage_idx]).item() if X_valids[usage_idx] else np.nan
            
        lidar_token  = nusc.get('sample', sample_token)['data']['LIDAR_TOP']
        lidar_record = nusc.get('sample_data', lidar_token)
        
        filename1 = os.path.join(intermediate_results_appearanceclustering_dir, os.path.basename(lidar_record['filename']).replace('.pcd.bin',f'__K-class-agnostic{str(K__class_agnostic).zfill(3)}_moving-fraction-thres0{str(int(10000*moving_fraction_thres)).zfill(4)}__usage-dict__class_agnostic.npy'))
        np.save(filename1, usage_dict__class_agnostic)
        
        filename2 = os.path.join(intermediate_results_appearanceclustering_dir, os.path.basename(lidar_record['filename']).replace('.pcd.bin',f'__K-class-agnostic{str(K__class_agnostic).zfill(3)}_moving-fraction-thres0{str(int(10000*moving_fraction_thres)).zfill(4)}__label-dict__class_agnostic.npy'))
        np.save(filename2, label_dict__class_agnostic)
        
    filename3 = os.path.join(intermediate_results_appearanceclustering_dir, f'velocityfraction-per-cluster__K-class-agnostic{str(K__class_agnostic).zfill(3)}_moving-fraction-thres0{str(int(10000*moving_fraction_thres)).zfill(4)}__class_agnostic.npy')
    np.save(filename3, velocityfraction_per_cluster)
    print(f'\nVelocity fractions are:\n{np.round(np.sort(np.array(list(velocityfraction_per_cluster.values()))), 2).tolist()}')
    
    filename4 = os.path.join(intermediate_results_appearanceclustering_dir, f'sizes-per-cluster__K-class-agnostic{str(K__class_agnostic).zfill(3)}_moving-fraction-thres0{str(int(10000*moving_fraction_thres)).zfill(4)}__class_agnostic.npy')
    np.save(filename4, sizes_per_cluster)
    print(f'\nCluster sizes are:\n{np.array(list(sizes_per_cluster.values()))[np.argsort(np.array(list(velocityfraction_per_cluster.values())))].tolist()}\n')
    
    
    # Step 4: Create pseudo-class labels (multi-class).
    print('Cluster appearance embeddings again for multi-class!\n')
    
    K__multi_class_list = hyperparameters['Step4__K__multi_class_list']   # Unit: 1.
    max_iterations      = hyperparameters['Step4__max_iterations']   # Unit: 1.
    num_init            = hyperparameters['Step4__num_init']   # Unit: 1.
    
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    
    labels__multi_class_Xpc_list = []
    for K__multi_class_Xpc in K__multi_class_list:
        cluster_instance = KMeans_Torch(init='k-means++', num_clusters=K__multi_class_Xpc, max_iterations=max_iterations, num_init=num_init, device=device)
        cluster_instance.fit(X=X_features.clone()[torch.tensor(use_cluster_bool__class_agnostic)], set_seed=True, random_seed=0)
        
        labels__multi_class_Xpc = cluster_instance.predict(X=X_features.clone()).cpu().float()
        labels__multi_class_Xpc[~torch.tensor(use_cluster_bool__class_agnostic)] = float('nan')

        labels__multi_class_Xpc_list.append(labels__multi_class_Xpc.clone().tolist())
        
        centers   = cluster_instance.centers.clone().cpu().numpy()
        filename5 = os.path.join(intermediate_results_appearanceclustering_dir, f'K-means-centers__K-multi-class{str(K__multi_class_Xpc).zfill(3)}_moving-fraction-thres0{str(int(10000*moving_fraction_thres)).zfill(4)}__multi-class.npy')
        np.save(filename5, centers)
        
        
    # Step 5: Save clustering results (multi-class).
    print('Save clustering result for multi-class!')
    
    for K__multi_class_Xpc, labels__multi_class_Xpc in zip(K__multi_class_list, labels__multi_class_Xpc_list):
        print(f'\nMulti-class with {K__multi_class_Xpc} classes:\n')
        
        for cnt, sample_token in enumerate(sample_tokens__train):
            if cnt%50==0:
                print(f'sample_token: {cnt+1}/{len(sample_tokens__train)}')
            
            label_dict__multi_class = {}
            for cluster_idx, usage_idx in enumerate(token_to_clusterids_dict[sample_token]):
                label_dict__multi_class[cluster_idx] = int(labels__multi_class_Xpc[usage_idx]) if not np.isnan(labels__multi_class_Xpc[usage_idx]) else np.nan
                
            lidar_token  = nusc.get('sample', sample_token)['data']['LIDAR_TOP']
            lidar_record = nusc.get('sample_data', lidar_token)
            
            filename6 = os.path.join(intermediate_results_appearanceclustering_dir, os.path.basename(lidar_record['filename']).replace('.pcd.bin',f'__K-multi-class{str(K__multi_class_Xpc).zfill(3)}_moving-fraction-thres0{str(int(10000*moving_fraction_thres)).zfill(4)}__label-dict__multi-class.npy'))
            np.save(filename6, label_dict__multi_class)
            
        velocityfraction_per_cluster, sizes_per_cluster = {}, {}
        for label in torch.unique(torch.tensor(labels__multi_class_Xpc)[~torch.isnan(torch.tensor(labels__multi_class_Xpc))]).int().tolist():
            label_bool                          = torch.tensor(labels__multi_class_Xpc)==label
            velocityfraction_per_cluster[label] = (((X_velocities[label_bool]>=velocity_thres) & (X_valids[label_bool])).sum() / label_bool.sum()).item()
            sizes_per_cluster[label]            = label_bool.sum().item()
            
        filename7 = os.path.join(intermediate_results_appearanceclustering_dir, f'velocityfraction-per-cluster__K-multi-class{str(K__multi_class_Xpc).zfill(3)}_moving-fraction-thres0{str(int(10000*moving_fraction_thres)).zfill(4)}__multi-class.npy')
        np.save(filename7, velocityfraction_per_cluster)
        print(f'\nVelocity fractions are:\n{np.round(np.sort(np.array(list(velocityfraction_per_cluster.values()))), 2).tolist()}')
        
        filename8 = os.path.join(intermediate_results_appearanceclustering_dir, f'sizes-per-cluster__K-multi-class{str(K__multi_class_Xpc).zfill(3)}_moving-fraction-thres0{str(int(10000*moving_fraction_thres)).zfill(4)}__multi-class.npy')
        np.save(filename8, sizes_per_cluster)
        print(f'\nCluster sizes are:\n{np.array(list(sizes_per_cluster.values()))[np.argsort(np.array(list(velocityfraction_per_cluster.values())))].tolist()}')
