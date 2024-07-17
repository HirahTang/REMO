import os
import argparse


# CUDA_VISIBLE_DEVICES=1 python finetune_model_scripts.py --input_model_file
# CUDA_VISIBLE_DEVICES=7 python finetune_model_scripts.py --input_model_file /share/project/sharefs-skfeng/Chemical_Reaction_Pretraining/pretrain/state2/mask_node_edge/modelrc.pth --adv
# CUDA_VISIBLE_DEVICES=7 python finetune_model_scripts.py --input_model_file /share/project/sharefs-skfeng/Chemical_Reaction_Pretraining/pretrain/state2/mlabel_adv_multitask/model_mlabes_rc.pth 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    # parser.add_argument("--pretrain_model", type=str, default="sce")
    # parser.add_argument('--input_model_file', type=str, help='input model', required=False, default=None)
    
    parser.add_argument('--input_model_files', nargs='+', help='pretrain_models', required=False, default=None)
    
    parser.add_argument("--output_dir", type=str, default="test_res")
    parser.add_argument("--JK", type=str, default="last")
    parser.add_argument('--adv', action='store_true', default=False, help='adversarial training')

    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')

    args = parser.parse_args()

    # input_model_file = args.input_model_file
    input_model_files = args.input_model_files
    adv = args.adv
    JK = args.JK



    epochs = args.epochs
    decay = args.decay

    task_lst = ['tox21', 'hiv',  'muv', 'bace', 'bbbp', 'toxcast', 'sider', 'clintox'] # 'pcba',
    # task_lst = ['muv', 'bace', 'bbbp', 'toxcast', 'sider', 'clintox'] # 'pcba',
    # task_lst = ['muv', 'toxcast']
    # task_lst = ['tox21']
    # task_lst = ['hiv',  'muv', 'bace', 'bbbp', 'toxcast', 'sider', 'clintox']
    task_lst = ['sider', 'bace', 'muv']

    task_lst = ['bbbp']
    seed_lst = [0, 1, 2]
    # seed_lst = [0]
    for input_model_file in input_model_files:
        for task in task_lst:
            for run_seed in seed_lst:
                model_dir = os.path.dirname(input_model_file)
                output_test_dir = f'{model_dir}/{task}'
                model_prefix = os.path.basename(input_model_file).split('.')[0]

                if task in ['muv']:
                    epochs = 50
                else:
                    epochs = 100
                
                if task in ['sider', 'bace']:
                    JK = 'MLP'
                else:
                    JK = 'last'

                base_cmd = f'python -u finetune_gcn.py --input_model_file {input_model_file} --dataset {task} --filename {output_test_dir} --runseed {run_seed} --device 0 --JK {JK} --epochs {epochs} --decay {decay} > {model_dir}/{model_prefix}_task_{task}_seed_{run_seed}_{JK}.log'

                if adv:
                    base_cmd = f'python -u finetune_gcn.py --input_model_file {input_model_file} --dataset {task} --filename {output_test_dir} --runseed {run_seed} --device 0  --adv --adv_lr 0.03 > {model_dir}/task_{task}_seed_{run_seed}_adv.log'


                print(base_cmd)
                os.system(base_cmd)