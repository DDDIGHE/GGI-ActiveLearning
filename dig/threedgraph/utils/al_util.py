import torch

def get_average_edge_per_molecule(batch_data, edge_index, edges_features, device):
    unique_molecules, nodes_per_molecule = batch_data.batch.unique(return_counts=True)
    maximum_node_index_per_molecule = torch.cumsum(nodes_per_molecule,0)
    maximum_node_index_per_molecule = [0] + maximum_node_index_per_molecule.tolist() #0 for using conditional
    
    print('unique_molecules: ',unique_molecules)
    print('print ', nodes_per_molecule)
    # print('Total nodes: ',sum(maximum_node_index_per_molecule))
    print('total nodes: ',maximum_node_index_per_molecule[-1])

    # print('maximum_node_index_per_molecule: ',maximum_node_index_per_molecule)
    num_edges_per_molecule = []
    for i in range(1, len(maximum_node_index_per_molecule)):
        edges_per_mol = edge_index[(edge_index>=maximum_node_index_per_molecule[i-1]) & (edge_index<maximum_node_index_per_molecule[i])]
        # print(f'max node index per mol: {edges_per_mol.max()}')

        num_edges_per_molecule.append(edges_per_mol.size(0))


    #spliting edges per molecule.
    edges_split = torch.split(edges_features, num_edges_per_molecule, dim=0)

    features = torch.tensor([]).to(device)
    for one_mol_edges in edges_split:
        # print('one_mol_edges.shape: ', one_mol_edges.shape)
        mean_edge = torch.mean(one_mol_edges, dim=0, keepdim=True)
        # print(mean_edge.shape)
        features = torch.cat((features, mean_edge), 0)
    return features


def get_average_edge_node_per_molecule(batch_data, edge_index, edges_features, nodes_features, device):
    unique_molecules, nodes_per_molecule = batch_data.batch.unique(return_counts=True)
    maximum_node_index_per_molecule = torch.cumsum(nodes_per_molecule,0)
    maximum_node_index_per_molecule = [0] + maximum_node_index_per_molecule.tolist() #0 for using conditional
    # 
    print('unique molecules: ',unique_molecules)
    print('nodes_per_molecule: ', nodes_per_molecule)
    print('Total nodes: ',sum(maximum_node_index_per_molecule))
    print('total nodes: ',maximum_node_index_per_molecule[-1])

    # print('maximum_node_index_per_molecule: ',maximum_node_index_per_molecule)
    num_edges_per_molecule = []
    for i in range(1, len(maximum_node_index_per_molecule)):
        edges_per_mol = edge_index[(edge_index>=maximum_node_index_per_molecule[i-1]) & (edge_index<maximum_node_index_per_molecule[i])]
        # print(f'max node index per mol: {edges_per_mol.max()}')

        num_edges_per_molecule.append(edges_per_mol.size(0))


    #spliting edges per molecule.
    edges_split = torch.split(edges_features, num_edges_per_molecule, dim=0)
    ####
    nodes_split = torch.split(nodes_features, nodes_per_molecule.tolist(), dim=0)
    # print('nodes_split: ', nodes_split)

    features = torch.tensor([]).to(device)
    for one_mol_edges, one_mol_nodes in zip(edges_split, nodes_split):
        # print('one_mol_edges.shape: ', one_mol_edges.shape)
        mean_edge = torch.mean(one_mol_edges, dim=0, keepdim=True)
        mean_node = torch.mean(one_mol_nodes, dim=0, keepdim=True)
        # print('one mol nodes: ', one_mol_nodes.shape)
        # print('mean node: ', mean_node.shape)
        # print('mean edge: ', mean_edge.shape)
        mean_node = mean_node.repeat(1,8)
        # print('mean node repeat: ', mean_node.shape)
        mean_edge_node = torch.cat((mean_edge, mean_node), 1)
        # print('mean edge cat: ', mean_edge.shape)
        features = torch.cat((features, mean_edge_node), 0)
    return features


def get_average_node_per_molecule(batch_data, nodes_features, device):
    unique_molecules, nodes_per_molecule = batch_data.batch.unique(return_counts=True)

    # print('unique molecules: ',unique_molecules)
    # print('nodes_per_molecule: ', nodes_per_molecule)


    
    nodes_split = torch.split(nodes_features, nodes_per_molecule.tolist(), dim=0)
    # print('nodes_split: ', nodes_split)

    features = torch.tensor([]).to(device)
    for one_mol_nodes in nodes_split:
        mean_node = torch.mean(one_mol_nodes, dim=0, keepdim=True)
        
        
        # print('mean node shape: ', mean_node.shape)
       
        features = torch.cat((features, mean_node), 0)

    # print(f'Features shape {features.shape}')
    return features

def disable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.eval()
            # print('Dropout: ', m.training)

def disable_dropouta(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            
            print('Dropout: ', m.training)