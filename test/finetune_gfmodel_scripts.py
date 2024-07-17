import os
import argparse


# CUDA_VISIBLE_DEVICES=1 python finetune_model_scripts.py --input_model_file
# CUDA_VISIBLE_DEVICES=7 python finetune_model_scripts.py --input_model_file /share/project/sharefs-skfeng/Chemical_Reaction_Pretraining/pretrain/state2/mask_node_edge/modelrc.pth --adv
# CUDA_VISIBLE_DEVICES=7 python finetune_model_scripts.py --input_model_file /share/project/sharefs-skfeng/Chemical_Reaction_Pretraining/pretrain/state2/mlabel_adv_multitask/model_mlabes_rc.pth 

# python ../test/finetune_gfmodel_scripts.py --input_model_files /data/protein/SKData/REMO/mlabes_identi_epoch_20_rc.pth --graphormer_config_yaml ../test/assets/graphormer_small.yaml --quick_task 0

# python ../test/finetune_gfmodel_scripts.py --input_model_files /data/protein/SKData/REMO/mlabes_identi_epoch_30_rc.pth --graphormer_config_yaml ../test/assets/graphormer_small.yaml --quick_task 0

# python ../test/finetune_gfmodel_scripts.py --input_model_files /data/protein/SKData/REMO/mlabes_identi_epoch_40_rc.pth --graphormer_config_yaml ../test/assets/graphormer_small.yaml --quick_task 0


# python ../test/finetune_gfmodel_scripts.py --input_model_files /data/protein/SKData/REMO/mlabes_identi_epoch_20_rc.pth --graphormer_config_yaml ../test/assets/graphormer_small.yaml --quick_task 1 --cu_id 6

# python ../test/finetune_gfmodel_scripts.py --input_model_files /data/protein/SKData/REMO/mlabes_identi_epoch_30_rc.pth --graphormer_config_yaml ../test/assets/graphormer_small.yaml --quick_task 1 --cu_id 6

# python ../test/finetune_gfmodel_scripts.py --input_model_files /data/protein/SKData/REMO/mlabes_identi_epoch_40_rc.pth --graphormer_config_yaml ../test/assets/graphormer_small.yaml --quick_task 1 --cu_id 7

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    # parser.add_argument("--pretrain_model", type=str, default="sce")
    # parser.add_argument('--input_model_file', type=str, help='input model', required=False, default=None)
    
    parser.add_argument('--input_model_files', nargs='+', help='pretrain_models', required=False, default=None)
    
    parser.add_argument("--graphormer_config_yaml", type=str, default="/test/assets/graphormer_standard.yaml")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument('--n_trials', type=int, default=30)

    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')

    parser.add_argument('--quick_task', type=int, default=0,
                        help='number of epochs to train (default: 100)')
    
    parser.add_argument('--cu_id', type=int, default=0,
                        help='number of epochs to train (default: 100)')

    args = parser.parse_args()

    # input_model_file = args.input_model_file
    input_model_files = args.input_model_files
    # adv = args.adv
    # JK = args.JK



    epochs = args.epochs
    # decay = args.decay
    patience = args.patience
    n_trials = args.n_trials
    graphormer_config_yaml = args.graphormer_config_yaml
    
    if args.quick_task:
        task_lst = ['bace', 'bbbp', 'toxcast', 'sider', 'clintox', 'tox21']
    else:
        task_lst = ['hiv',  'muv']

    # task_lst = ['tox21', 'hiv',  'muv', 'bace', 'bbbp', 'toxcast', 'sider', 'clintox'] # 'pcba',
    # task_lst = ['muv', 'bace', 'bbbp', 'toxcast', 'sider', 'clintox'] # 'pcba',
    # task_lst = ['muv', 'toxcast']
    # task_lst = ['tox21']
    # task_lst = ['hiv',  'muv', 'bace', 'bbbp', 'toxcast', 'sider', 'clintox']
    # task_lst = ['sider', 'bace', 'muv']
    # task_lst = ['bbbp']

    # task_lst = ['bbbp']
    seed_lst = [0, 1, 2]
    # seed_lst = [0]
    cu_id = args.cu_id
    if input_model_files is None: # train from scratch
        for task in task_lst:
            for run_seed in seed_lst:
                model_prefix = 'wo_pretrain'
                if task in ['muv']:
                    epochs = 50
                else:
                    epochs = 100
                
                base_cmd = f'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/home/fengshikun/miniconda3/envs/MOLuni/lib:$LD_LIBRARY_PATH; CUDA_VISIBLE_DEVICES={cu_id} python -u finetune_graphformer_optuna_molnet.py --dataset {task} --graphormer_config_yaml {graphormer_config_yaml} --patience {patience} --n_trials {n_trials} --epochs {epochs} --runseed {run_seed} > {model_prefix}_task_{task}_seed_{run_seed}.log'
                if task in ['hiv', 'muv']:
                    cu_id += 1
                    base_cmd += ' 2>&1 &'
                # if adv:
                #     base_cmd = f'python -u finetune_gcn.py --input_model_file {input_model_file} --dataset {task} --filename {output_test_dir} --runseed {run_seed} --device 0  --adv --adv_lr 0.03 > {model_dir}/task_{task}_seed_{run_seed}_adv.log'

                print(base_cmd)
                # os.system(base_cmd)
            print('\n\n')
        exit(0)

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
                
                # if task in ['sider', 'bace']:
                #     JK = 'MLP'
                # else:
                #     JK = 'last'

                # base_cmd = f'python -u finetune_gcn.py --input_model_file {input_model_file} --dataset {task} --filename {output_test_dir} --runseed {run_seed} --device 0 --JK {JK} --epochs {epochs} --decay {decay} > {model_dir}/{model_prefix}_task_{task}_seed_{run_seed}_{JK}.log'

                base_cmd = f'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/home/fengshikun/miniconda3/envs/MOLuni/lib:$LD_LIBRARY_PATH; CUDA_VISIBLE_DEVICES={cu_id} python -u finetune_graphformer_optuna_molnet.py --input_model_file {input_model_file} --dataset {task} --graphormer_config_yaml {graphormer_config_yaml} --patience {patience} --n_trials {n_trials} --epochs {epochs} --runseed {run_seed} > {model_prefix}_task_{task}_seed_{run_seed}.log'
                if task in ['hiv', 'muv']:
                    cu_id += 1
                    base_cmd += ' 2>&1 &'
                # if adv:
                #     base_cmd = f'python -u finetune_gcn.py --input_model_file {input_model_file} --dataset {task} --filename {output_test_dir} --runseed {run_seed} --device 0  --adv --adv_lr 0.03 > {model_dir}/task_{task}_seed_{run_seed}_adv.log'


                print(base_cmd)
                os.system(base_cmd)
            print('\n\n')