from tqdm import tqdm
import torch
from torch.optim import Adam


from torch_geometric.loader import DataLoader

from .util import get_average_node_per_molecule

bce_loss = torch.nn.BCELoss()

def train_discriminator(wandb, model, discriminator, labeled_dataloader, unlabeled_dataloader, device, num_epochs=32):
    
    optim_discriminator = Adam(discriminator.parameters(), lr=5e-4, weight_decay=0)

    model.eval()
    labeled_iterator = iter(labeled_dataloader)
    
    for epoch in range(num_epochs):
        loss_accum = 0
        for step, data_unlabeled in enumerate(tqdm(unlabeled_dataloader)):

            try:
                data_labeled = next(labeled_iterator)
            except StopIteration:
                labeled_iterator = iter(labeled_dataloader)
                data_labeled = next(labeled_iterator)

            data_labeled = data_labeled.to(device)
            data_unlabeled = data_unlabeled.to(device)

            labeled_edge_index, labeled_feature_dict, _ = model(data_labeled)
            unlabeled_edge_index, unlabeled_feature_dict, _ = model(data_unlabeled)

            labeled_node_features = labeled_feature_dict['node_features']
            unlabeled_node_features = unlabeled_feature_dict['node_features']

            labeled_av_node_features = get_average_node_per_molecule(data_labeled, labeled_node_features[-1].detach(), device)
            unlabeled_av_node_features = get_average_node_per_molecule(data_unlabeled, unlabeled_node_features[-1].detach(), device)

            # print(f'shape of lab : {labeled_av_node_features.size()}')
            # print(f'shape of unlab : {unlabeled_av_node_features.size()}')
            #get disc predictions

            lab_pred = discriminator(labeled_av_node_features)
            unlab_pred = discriminator(unlabeled_av_node_features)

            # print(f'shape of lab : {lab_pred}')
            # print(f'shape of unlab : {unlab_pred}')
            
            lab_gt = torch.ones(lab_pred.size()).to(device)
            unlab_gt = torch.zeros(unlab_pred.size()).to(device)

            dsc_loss = bce_loss(lab_pred, lab_gt) + bce_loss(unlab_pred, unlab_gt)

            optim_discriminator.zero_grad()
            dsc_loss.backward()
            optim_discriminator.step()

            loss_accum += dsc_loss.detach().cpu().item()
        wandb.log({'dis_loss': loss_accum/(step+1)}, step = epoch)



    
