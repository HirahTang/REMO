import torch
import sys
# sys.path.append(sys.path[0]+'/..')
sys.path.insert(0,'..')

import yaml

from models.graphormer import GraphormerEncoder
from datas.graphormer_data import MyZINC, GraphormerREACTDataset, BatchedDataDataset
from tqdm import tqdm

from data_analysis.USPTO_CONFIG import USPTO_CONFIG
from data_processing import ReactionDataset


class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

def main():
    # init model
    device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
    config_file = 'assets/graphormer.yaml'
    with open(config_file, 'r') as cr:
        model_config = yaml.safe_load(cr)
    model_config = Struct(**model_config)
    task_info = {'mask': True, 'indenti': True, 'mlabel_task_vocab_size': 2458, 'finetune': False, 'task_num': 0}
    task_info = Struct(**task_info)
    encoder = GraphormerEncoder(model_config, task_info).to(device)
    # init dataset 
    # root = '/share/project/tanghan/sharefs-hantang/ZINC/dataset'
    # inner_dataset = MyZINC(root=root)
    # train_set = MyZINC(root=root, split="train")
    # valid_set = MyZINC(root=root, split="val")
    # test_set = MyZINC(root=root, split="test")
    seed = 0

    reaction_dataset = ReactionDataset(data_path=USPTO_CONFIG.dataset, atom_vocab=USPTO_CONFIG.atom_vocab_file)

    reaction_dataset_size = len(reaction_dataset)
    train_size = int(reaction_dataset_size * 0.8)
    val_size = reaction_dataset_size - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(reaction_dataset, [train_size, val_size])


    data_set = GraphormerREACTDataset(
                    None,
                    seed,
                    None,
                    None,
                    None,
                    train_dataset,
                    val_dataset,
                    # test_set=val_dataset,
                )
    batched_data = BatchedDataDataset(
            data_set.train_data,
            max_node=model_config.max_nodes,
            multi_hop_max_dist=model_config.multi_hop_max_dist,
            spatial_pos_max=model_config.spatial_pos_max,
        )

    # init dataloader
    data_loader = torch.utils.data.DataLoader(batched_data, batch_size=4, shuffle=True, num_workers = 1, collate_fn = batched_data.collater)
    # forward
    for step, batch in enumerate(tqdm(data_loader, desc="Iteration")):
        for k in batch:
            if k not in ['mask_atom_indices', 'reaction_centre', 'mlabes']:
                batch[k] = batch[k].to(device)
        # batch = batch.to(device)
        node_rep = encoder(batch)
        print('finish')

    pass


if __name__ == "__main__":
    main()