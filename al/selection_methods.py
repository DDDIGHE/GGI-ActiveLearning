import time
import numpy
from tqdm import tqdm
import random

import sys

from scipy.spatial.distance import cdist
from fastdist import fastdist
import numpy as np
from qpsolvers import solve_qp



from torchmetrics.functional import pairwise_cosine_similarity, pairwise_euclidean_distance

import torch
from torch_geometric.loader import DataLoader
from torch_scatter import scatter_mean

from .sampler import SubsetSequentialSampler
from .util import get_average_edge_per_molecule, get_average_edge_node_per_molecule, get_average_node_per_molecule



def enable_dropout(model):
    for m in model.modules():
        if m.__class__.__name__.startswith('ConcreteDropout'):
            m.train()
def random_sel(labeled_indices, unlabeled_indices, k=1500):
    return random.sample(unlabeled_indices, k) + labeled_indices

def mc_dropout(model, unlabeled_indices, data, device, k=1500, with_diversity=False):

    time_start = time.time()
    pred_matrix = torch.tensor([]).to(device)

    for i in range(20):
        unlabeled_loader = DataLoader(data, batch_size=64, 
                            sampler=SubsetSequentialSampler(unlabeled_indices), # more convenient if we maintain the order of subset
                            pin_memory=True, drop_last=False)

        print('Len_unlabloader: ', len(unlabeled_loader))
        
        #set model to evaluation mode
        model.eval()
        enable_dropout(model)

        with torch.cuda.device(device):
            preds = torch.tensor([]).to(device)

        with torch.no_grad():
            for step, batch_data in enumerate(tqdm(unlabeled_loader)):
                batch_data = batch_data.to(device)
            
                with torch.no_grad():
                    _, _, out, _ = model(batch_data)
                    # print(out.shape)
                preds = torch.cat([preds, out.detach_()], dim=0)

        pred_matrix = torch.cat([pred_matrix, preds], dim=1)
        # print(pred_matrix[:10, :])
    
    
    variance = torch.var(pred_matrix, dim=1)
    if with_diversity:
        return variance

    arg = torch.argsort(variance, descending=True).cpu()
    return torch.tensor(unlabeled_indices)[arg[:k]], (time.time()-time_start)/60

##uncertainty rep using euclidean distance

def compute_USR(mol_pos):
    # Function to calculate four moments for a given set of values
    def calculate_moments(values):
        mean_value = values.mean()
        variance = values.var()
        skewness = (torch.abs((values - mean_value) ** 3)).mean().sqrt()
        kurtosis = ((values - mean_value) ** 4).mean().sqrt()
        return [mean_value, variance, skewness, kurtosis]

    # Function to calculate the angle between two vectors
    def calculate_angle(a, b):
        cos_angle = torch.dot(a, b) / (torch.norm(a) * torch.norm(b))
        angle = torch.acos(torch.clamp(cos_angle, -1, 1))
        return angle

    # Calculate the centroid
    centroid = mol_pos.mean(dim=0)

    # Find the first farthest point from the centroid
    distances_to_centroid = torch.norm(mol_pos - centroid, dim=1)
    farthest_point = mol_pos[torch.argmax(distances_to_centroid)]
    closest_point = mol_pos[torch.argmin(distances_to_centroid)]
    farfromfarthest_point = mol_pos[torch.argmax(farthest_point)]
    
    # Calculate the reference vector from centroid to the farthest point
    reference_vector = farthest_point - centroid
    reference_vector1 = closest_point - centroid
    reference_vector2 = farfromfarthest_point - centroid

    # Compute angles with the reference vector for atoms to centroid
    angles1 = [calculate_angle(reference_vector, pt - centroid) for pt in mol_pos]
    angles11 = [calculate_angle(reference_vector1, pt - centroid) for pt in mol_pos]
    angles12 = [calculate_angle(reference_vector2, pt - centroid) for pt in mol_pos]
    
    # Compute angles with the reference vector for all possible atom pairs
    angles2 = []
    for i in range(len(mol_pos)):
        for j in range(len(mol_pos)):
            if i != j:
                angle = calculate_angle(reference_vector, mol_pos[j] - mol_pos[i])
                angles2.append(angle)
    angles21 = []
    for i in range(len(mol_pos)):
        for j in range(len(mol_pos)):
            if i != j:
                angle = calculate_angle(reference_vector1, mol_pos[j] - mol_pos[i])
                angles21.append(angle)
    angles22 = []
    for i in range(len(mol_pos)):
        for j in range(len(mol_pos)):
            if i != j:
                angle = calculate_angle(reference_vector2, mol_pos[j] - mol_pos[i])
                angles22.append(angle)

    # Calculate moments for distances from 4 reference points
    distances_to_point1 = torch.norm(mol_pos - farthest_point, dim=1)
    farthest_point2 = mol_pos[torch.argmax(distances_to_point1)]
    
    distances_to_point2 = torch.norm(mol_pos - centroid, dim=1)
    farthest_point3 = mol_pos[torch.argmin(distances_to_point2)]

    points = [centroid, farthest_point, farthest_point2, farthest_point3]
    moments_all = []
    for point in points:
        distances = torch.norm(mol_pos - point, dim=1)
        moments_all.extend(calculate_moments(distances))
    #print("calculate_moments(distances):", torch.tensor(calculate_moments(distances)))

    # Add moments for both sets of angles
    moments_all.extend(calculate_moments(torch.tensor(angles1)))
    moments_all.extend(calculate_moments(torch.tensor(angles11)))
    moments_all.extend(calculate_moments(torch.tensor(angles12)))
    moments_all.extend(calculate_moments(torch.tensor(angles2)))
    moments_all.extend(calculate_moments(torch.tensor(angles21)))
    moments_all.extend(calculate_moments(torch.tensor(angles22)))
    #print("Shape of moments_all:", torch.tensor(moments_all).shape)

    return torch.tensor(moments_all)


def pairwise_usr_similarity(matrix1, matrix2=None):
    if matrix2 is None:
        matrix2 = matrix1
        
    # Compute distances
    distances = cdist(matrix1.cpu().numpy(), matrix2.cpu().numpy(), 'cityblock') / 4.0 + 1.0
    similarities = 1.0 / distances
    
    # Convert similarities to a torch tensor
    tensor_similarities = torch.tensor(similarities, dtype=torch.float32)
    
    # Print the first 5x5 block of tensor_similarities
    #print("First 5x5 block of tensor_similarities:")
    #print(tensor_similarities[:5, :5])
    
    # Check if tensor_similarities is symmetric
    #if not torch.allclose(tensor_similarities, tensor_similarities.t(), atol=1e-8):
    #    print("Warning: tensor_similarities is not symmetric!")
    #    # Output the elements at the non-symmetric positions
    #    for i in range(5):  # check the first 5 elements
    #        val1 = tensor_similarities[i, i+1].item()
    #        val2 = tensor_similarities[i+1, i].item()
    #        print(f"Element at ({i}, {i+1}): {val1}, Element at ({i+1}, {i}): {val2}")
            
    #for i in range(5):  # check the first 5 elements
    #        val1 = tensor_similarities[i, i+1].item()
    #        val2 = tensor_similarities[i+1, i].item()
    #        print(f"Element at ({i}, {i+1}): {val1}, Element at ({i+1}, {i}): {val2}")

    # Check if tensor_similarities is positive definite
    #e, _ = torch.linalg.eigh(tensor_similarities)
    #if torch.all(e > 0):
    #    print("tensor_similarities is positive definite!")
    #else:
    #    print("Warning: tensor_similarities is not positive definite!")
    #    non_positive_eigenvalues = e[e <= 0].tolist()
    #    print(f"Non-positive eigenvalues: {non_positive_eigenvalues}")
    
    # Convert similarities to a torch tensor and perform the 1-x operation
    tensor_similarities = 1 - torch.tensor(similarities, dtype=torch.float32)

    return tensor_similarities




def compute_uncertainty_diversity(model, labeled_indices, unlabeled_indices, data, device, k=1500):

    time_start = time.time()
    unlabeled_loader = DataLoader(data, batch_size=64, 
                        sampler=SubsetSequentialSampler(labeled_indices+unlabeled_indices), # more convenient if we maintain the order of subset
                        pin_memory=True, drop_last=False)

    print('Len_unlabloader: ', len(unlabeled_loader))
    
    #set model to evaluation mode
    model.eval()

    with torch.cuda.device(device):
        #features = torch.tensor([]).to(device)
        features1 = torch.tensor([]).to(device)

    with torch.no_grad():
        for step, batch_data in enumerate(tqdm(unlabeled_loader)):
            batch_data = batch_data.to(device)
          
            with torch.no_grad():
                
                #_, feature_dict, _, _ = model(batch_data)
                # edge_features = feature_dict['edge_features']
                z, pos, batch = batch_data.z, batch_data.pos, batch_data.batch
                #node_features = feature_dict['node_features']

            #av_node_features = get_average_node_per_molecule(batch_data, node_features[-1], device)
            #features = torch.cat((features, av_node_features), 0)
            usr_features = []
            for mol_idx in batch.unique():
                mol_pos = pos[batch == mol_idx]
                usr_features.append(compute_USR(mol_pos))
            #print("Shape of usr_features:", usr_features.shape)
            usr_features = torch.stack(usr_features).to(device)
            features1 = torch.cat((features1, usr_features), 0)
            
    #print("Shape of features:", features.shape)

    lab_predictions = features1[:len(labeled_indices)]
    unlab_predictions = features1[len(labeled_indices):]

    print(f'Train_shape: {lab_predictions.shape}:::: un_shape: {unlab_predictions.shape}')
    
   
    #maximize diversity
    diversity_matrix = pairwise_usr_similarity(unlab_predictions)
    #if not torch.allclose(diversity_matrix, diversity_matrix.t(), atol=1e-8):
    #    print("Warning: tensor_similarities is not symmetric!")
    diversity_matrix = diversity_matrix/diversity_matrix.max()
    diversity_matrix.fill_diagonal_(0.0)
    #print("First 5x5 block of diversity_matrix:")
    #print(diversity_matrix[:5, :5])
    
    # Check if diversity_matrix is symmetric
    #if not torch.allclose(diversity_matrix, diversity_matrix.t(), atol=1e-8):
    #    print("Warning: diversity_matrix is not symmetric!")
    #    # Output the elements at the non-symmetric positions
    #    for i in range(5):  # check the first 5 elements
    #        val1 = diversity_matrix[i, i+1].item()
    #        val2 = diversity_matrix[i+1, i].item()
    #        print(f"Element at ({i}, {i+1}): {val1}, Element at ({i+1}, {i}): {val2}")
            
    #for i in range(5):  # check the first 5 elements
    #        val1 = diversity_matrix[i, i+1].item()
    #        val2 = diversity_matrix[i+1, i].item()
    #        print(f"Element at ({i}, {i+1}): {val1}, Element at ({i+1}, {i}): {val2}")

    # Check if diversity_matrix is positive definite
    #e, _ = torch.linalg.eigh(diversity_matrix)
    #if torch.all(e > 0):
    #    print("diversity_matrix is positive definite!")
    #else:
    #    print("Warning: diversity_matrix is not positive definite!")
    #    non_positive_eigenvalues = e[e <= 0].tolist()
    #    print(f"Non-positive eigenvalues: {non_positive_eigenvalues}")

    #because the solver minimizes the qp problem
    diversity_matrix = 300-300*diversity_matrix
    diversity_matrix = diversity_matrix.cpu().numpy()
    print('diversity_matrix shape: ', diversity_matrix.shape)


    uncertainty = mc_dropout(model, unlabeled_indices, data, device, k=1500, with_diversity=True)
    uncertainty = uncertainty/uncertainty.max()
    uncertainty = 1-uncertainty
    uncertainty = uncertainty.cpu().numpy()
    #uncertainty = -uncertainty


    ### setting upper bound lower bounds and constraints
    lb = np.zeros(len(unlabeled_indices))
    ub = np.ones(len(unlabeled_indices))

    A = np.ones(len(unlabeled_indices))
    b = 1.0 * np.array([k])
    
    # un_rep = uncertainty 

    print(f'Solving qp....')
    res = solve_qp(P=diversity_matrix, q=uncertainty, G=None, A=A, b=b, lb=lb, ub=ub, solver='osqp', verbose=True)
  
    arg = np.argsort(res)
    query_indices = np.array(unlabeled_indices)[arg[:k]]
    return labeled_indices+query_indices.tolist(), (time.time()-time_start)//60

def get_loss(model, lossnet, unlabeled_loader, device):
    model.eval()
    lossnet.eval()
    with torch.cuda.device(device):
        loss = torch.tensor([]).to(device)

    with torch.no_grad():
        for step, batch_data in enumerate(tqdm(unlabeled_loader)):
            batch_data = batch_data.to(device)
            with torch.no_grad():
                _, feature_dict, _, _ = model(batch_data)
                node_features = feature_dict['node_features']

            features_list = []
            for n_feature in node_features:
                av_node_features = get_average_node_per_molecule(batch_data, n_feature, device)
                features_list.append(av_node_features)
           
            lossnet_out = lossnet(features_list)                
            lossnet_out = lossnet_out.view(lossnet_out.size(0))
            loss = torch.cat((loss, lossnet_out), 0)
    
    return torch.abs(loss)


def lloss_sel(model, lossnet, unlabeled_indices, data, device, k=1500):
    time_start = time.time()
    unlabeled_loader = DataLoader(data, batch_size=64, 
                                    sampler=SubsetSequentialSampler(unlabeled_indices), 
                                    pin_memory=True)

    loss = get_loss(model, lossnet, unlabeled_loader, device)
    arg = torch.argsort(loss, descending=True).cpu()
    print(arg)
    print('Querying lloss   ----- Len : ', len(arg))
    print('len data: ', len(unlabeled_indices))
    return torch.tensor(unlabeled_indices)[arg[:k]], (time.time()-time_start)/60


##node features
def coreset(model, labeled_indices, unlabeled_indices, data, device, k=1500):

    time_start = time.time()
    unlabeled_loader = DataLoader(data, batch_size=8, 
                        sampler=SubsetSequentialSampler(labeled_indices+unlabeled_indices), # more convenient if we maintain the order of subset
                        pin_memory=True, drop_last=False)

    print('Len_unlabloader: ', len(unlabeled_loader))
    
    #set model to evaluation mode
    model.eval()

    with torch.cuda.device(device):
        features = torch.tensor([]).to(device)

    with torch.no_grad():
        i=0
        for step, batch_data in enumerate(tqdm(unlabeled_loader)):
            batch_data = batch_data.to(device)
          
            with torch.no_grad():
                _, feature_dict, _, _ = model(batch_data)
                node_features = feature_dict['node_features']

            av_node_features = get_average_node_per_molecule(batch_data, node_features[-1], device)
            features = torch.cat((features, av_node_features), 0)
        
        
    feat = features.detach().cpu().numpy()
    print(feat.shape)

    train_predictions = feat[:len(labeled_indices)]
    predictions = feat[len(labeled_indices):]

    print(f'Train_shape: {train_predictions.shape}:::: un_shape: {predictions.shape}')
    '''
    This functions takes feature values of labeled dataset and unlabeled dataset and return unlabeled indices based on the coreset method.
    '''
    
    
    subset_indices = numpy.array(unlabeled_indices)

    query_indices = []
    unlabeled_pairwise_distance = fastdist.matrix_pairwise_distance(predictions, metric=fastdist.euclidean, metric_name="euclidean", return_matrix=True)
    distance = cdist(predictions, train_predictions, metric="euclidean")
    # unlabeled_pairwise_distance = fastdist.matrix_pairwise_distance(predictions[subset_indices], metric=fastdist.euclidean, metric_name="euclidean", return_matrix=True)
    # distance = cdist(predictions[subset_indices], train_predictions, metric="euclidean")
    print(f'Samples to select {k}')
    for i in range(0, k):
        t1 = time.time()
        min_distances = numpy.min(distance, axis=1)
        max_idx = numpy.argmax(min_distances)
        print(f'selecting {i}th sample of index {subset_indices[max_idx]}')
        if subset_indices[max_idx] in labeled_indices:
            print('Already in labeled;;;;stop stop')
            break
        query_indices.append(subset_indices[max_idx])
        # print(distance.shape, " ", unlabeled_pairwise_distance[max_idx].shape, end = "\r")
        distance = numpy.append(distance, numpy.array([unlabeled_pairwise_distance[max_idx]]).T, 1)
        distance = numpy.delete(distance, max_idx, 0)
        unlabeled_pairwise_distance = numpy.delete(unlabeled_pairwise_distance, max_idx, 0)
        unlabeled_pairwise_distance = numpy.delete(unlabeled_pairwise_distance, max_idx, 1)
        subset_indices = numpy.delete(subset_indices, max_idx) 
        print(f'Time taken for selecting {i+1}th sample: {(time.time() - t1)} secs\n') 
    
    return labeled_indices+query_indices, (time.time()-time_start)//60
