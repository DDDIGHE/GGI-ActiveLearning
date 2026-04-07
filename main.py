import argparse
import wandb
import torch
import numpy as np
import os
import random
from torch_geometric.loader import DataLoader


from dig.threedgraph.dataset import QM93D
from dig.threedgraph.dataset import MD17
from dig.threedgraph.method import SphereNet #SchNet, DimeNetPP, ComENet
from dig.threedgraph.method import run
from dig.threedgraph.evaluation import ThreeDEvaluator

from al import *

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--selection_method', default='random', type=str,  help='selection method')
    parser.add_argument('--dataset', default='qm9_pyg_25k_0.pt', type=str,  help='.pt dataset')
    parser.add_argument('--device', default=0, type=int,  help='GPU Device')
    parser.add_argument('--cycle', default=0, type=int,  help='AL Cycle')
    parser.add_argument('--expt', default=1, type=int,  help='Eperiment number')
    parser.add_argument('--ADDENDUM', default=1500, type=int,  help='No of data to add')
    #wandb
    parser.add_argument('--project', default='al_unc_div', type=str, help='wandb project name')
    parser.add_argument('--entity', default='your_entity', type=str, help='wandb entity name')
    parser.add_argument('--run_name', default='try', type=str, help='wandb run name')
    return parser.parse_args()

if __name__ == '__main__':

    setup_seed(8848)

    opt = parse_opt()

    selection_method = opt.selection_method
    print(f'Selection method: {selection_method}')
    assert selection_method in ['vae', 'coreset', 'random', 'lloss', 'rep', 'dis', 'dropout', 'unc_div_rep', 'unc_div', 'mcdrop']
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = opt.device
    print(f"Using device: {device}")
    cycle = opt.cycle
    expt = opt.expt
    ADDENDUM = opt.ADDENDUM
    dataset = opt.dataset
    print(f'Dataset: {dataset}')

    #wandb
    #project = opt.project
    #entity = opt.entity
    #run_name = opt.run_name

    #wandb.init(project=project, entity=entity, name=run_name)

    dataset = QM93D(root='dataset/', processed_fn=dataset)
    target = 'mu' # choose from: mu, alpha, homo, lumo, r2, zpve, U0, U, H, G, Cv
    print(f'Target prediction: {target}')
    dataset.data.y = dataset.data[target]

    print(f'Length of entire dataset: {len(dataset.data.y)}')

    split_idx = dataset.get_idx_split(len(dataset.data.y), train_size=25000,valid_size=10000, 
                                      seed=42, method= selection_method, cycle=cycle, expt=expt, ADDENDUM=ADDENDUM)

    train_dataset, valid_dataset, test_dataset, unlab_dataset = dataset[split_idx['train']], dataset[split_idx['valid']], dataset[split_idx['test']], dataset[split_idx['unlabeled']]
    
    ## training on full dataset##
    # train_dataset, valid_dataset, test_dataset, unlab_dataset = dataset[torch.tensor(split_idx['train'].tolist()+split_idx['unlabeled'].tolist())], dataset[split_idx['valid']], dataset[split_idx['test']], dataset[split_idx['unlabeled']]
    print('train, validaion, test, unlab_dataset:', len(train_dataset), len(valid_dataset), len(test_dataset), len(unlab_dataset))
    print('train: ', train_dataset)
    # print(len(split_idx['train'].tolist()))

    model = SphereNet(energy_and_force=False, cutoff=5.0, num_layers=4, 
        hidden_channels=128, out_channels=1, int_emb_size=64, 
        basis_emb_size_dist=8, basis_emb_size_angle=8, basis_emb_size_torsion=8, out_emb_channels=256, 
        num_spherical=3, num_radial=6, envelope_exponent=5, 
        num_before_skip=1, num_after_skip=2, num_output_layers=3, use_node_features=True, droprate=0.2
        )
    model = model.to(device)
    loss_func = torch.nn.L1Loss()
    evaluation = ThreeDEvaluator()
 

    lossnet = None
    lossnet_loss_func = None
    if selection_method == 'lloss':
        lossnet = LossNet()
        lossnet = lossnet.to(device)
        lossnet_loss_func = LossPredLoss

    run3d = run()
    run3d.run(wandb, device, train_dataset, valid_dataset, test_dataset, model, loss_func, evaluation,
          method=selection_method,lossnet=lossnet, lossnet_loss_func=lossnet_loss_func,
          epochs=150, batch_size=32,
          vt_batch_size=64, lr=0.0005, lr_decay_factor=0.5, 
          lr_decay_step_size=15, expt=expt, cycle=cycle)

    if selection_method == 'random':
        query_indices = random_sel(split_idx['train'].tolist(), split_idx['unlabeled'].tolist(), k=ADDENDUM)
        # print('random len query: ', len(query_indices))
        np.save(f"runs/{selection_method}/run{expt}/init_set.npy", np.asarray(query_indices))

    
    if selection_method == 'mcdrop':
        print(f'{selection_method}')
        model.eval()
        query_indices, time_taken = mc_dropout(model, split_idx['unlabeled'].tolist(),dataset[:25000], 
                            device=device, k=ADDENDUM)
        np.save(f"runs/{selection_method}/run{expt}/init_set.npy", np.asarray(query_indices.tolist()+split_idx['train'].tolist()))

        if not os.path.exists(f'runs/{selection_method}/run{expt}/time.txt'):
            with open(os.path.join(f'runs/{selection_method}/run{expt}', f'time.txt'), 'w') as f:
                f.write(f'CYCLE: {cycle} AL Time: {time_taken}\n')
                
        else:
            with open(os.path.join(f'runs/{selection_method}/run{expt}', f'time.txt'), 'a') as f:
                f.write(f'CYCLE: {cycle}   AL Time: {time_taken}\n')
    
    if selection_method == 'coreset':
        print('coreset')
        model.eval()
        query_indices, time_taken = coreset(model, split_idx['train'].tolist(), split_idx['unlabeled'].tolist(),dataset[:25000], 
                            device=device, k=ADDENDUM)
        np.save(f"runs/{selection_method}/run{expt}/init_set.npy", np.asarray(query_indices))

        if not os.path.exists(f'runs/{selection_method}/run{expt}/time.txt'):
            with open(os.path.join(f'runs/{selection_method}/run{expt}', f'time.txt'), 'w') as f:
                f.write(f'CYCLE: {cycle} AL Time: {time_taken}\n')
                
        else:
            with open(os.path.join(f'runs/{selection_method}/run{expt}', f'time.txt'), 'a') as f:
                f.write(f'CYCLE: {cycle}   AL Time: {time_taken}\n')

   
    ### Uncertainity and Representativeness ###
    if selection_method == 'unc_div':
        print('uncertainty_diversity')
        model.eval()
        query_indices, time_taken = compute_uncertainty_diversity(model, split_idx['train'].tolist(), split_idx['unlabeled'].tolist(),dataset[:25000], 
            device=device, k=ADDENDUM)
        print(f'Leght of eleca fvaesrf  {len(query_indices)}')
        np.save(f"runs/{selection_method}/run{expt}/init_set.npy", np.asarray(query_indices))

        if not os.path.exists(f'runs/{selection_method}/run{expt}/time.txt'):
            with open(os.path.join(f'runs/{selection_method}/run{expt}', f'time.txt'), 'w') as f:
                f.write(f'CYCLE: {cycle} AL Time: {time_taken}\n')
                
        else:
            with open(os.path.join(f'runs/{selection_method}/run{expt}', f'time.txt'), 'a') as f:
                f.write(f'CYCLE: {cycle}   AL Time: {time_taken}\n')
    
    
    
    if selection_method == 'lloss':
        model.eval()
        lossnet.eval()
        query_indices, time_taken = lloss_sel(model, lossnet, split_idx['unlabeled'].tolist(),dataset[:25000], 
                        device=device, k=ADDENDUM)

        np.save(f"runs/{selection_method}/run{expt}/init_set.npy", np.asarray(query_indices.tolist()+split_idx['train'].tolist()))

        if not os.path.exists(f'runs/{selection_method}/run{expt}/time.txt'):
            with open(os.path.join(f'runs/{selection_method}/run{expt}', f'time.txt'), 'w') as f:
                f.write(f'CYCLE: {cycle} AL Time: {time_taken}\n')
                
        else:
            with open(os.path.join(f'runs/{selection_method}/run{expt}', f'time.txt'), 'a') as f:
                f.write(f'CYCLE: {cycle}   AL Time: {time_taken}\n')


