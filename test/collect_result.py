import os
import argparse
import numpy as np
task_lst = ['tox21', 'hiv',  'muv', 'bace', 'bbbp', 'toxcast', 'sider', 'clintox']

task_lst = ['bbbp', 'tox21', 'toxcast', 'sider', 'clintox', 'muv', 'hiv', 'bace']

# task_lst = ['sider', 'muv', 'bace']

# task_lst = ['bbbp']

def get_final_res(log_file):
    with open(log_file, 'r') as lr:
        log_lines = [line.strip() for line in lr.readlines()]
        res_line = ''
        for line in log_lines:
            if line.startswith('train: '):
                res_line = line
        test_res = float(res_line.split()[-1])
    return test_res

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='collect results')
    parser.add_argument("--adv", type=bool, default=False)
    parser.add_argument('--test_dir', type=str, help='test directionary', required=False, default=None)
    parser.add_argument('--model_prefix', type=str, help='test model prefix', required=False, default='')

    parser.add_argument('--suffix', type=str, help='log suffix', required=False, default='')

    args = parser.parse_args()
    test_dir = args.test_dir
    model_prefix = args.model_prefix

    log_suffix = args.suffix
    seed_num = 3
    
    res_file = 'collect_res.txt'
    res_lst = []

    task_res_lst = []
    for task in task_lst:
        test_res = []
        for seed in range(seed_num):
            if len(model_prefix):
                log_file = f'{model_prefix}_task_{task}_seed_{seed}{log_suffix}.log'
            else:
                log_file = f'task_{task}_seed_{seed}{log_suffix}.log'
            if args.adv:
                if len(model_prefix):
                    log_file = f'{model_prefix}_task_{task}_seed_{seed}_adv.log'
                else:
                    log_file = f'task_{task}_seed_{seed}_adv.log'
            log_path = os.path.join(test_dir, log_file)
            test_res.append(get_final_res(log_path))
        task_res_lst.append(np.mean(test_res))
        res_lst.append(f'{np.mean(test_res) * 100:.2f}({np.var(test_res) * 100:.3f})')
        print(f'{task}: {np.mean(test_res) * 100:.2f}({np.var(test_res) * 100:.3f})')
    
    res_lst.append(f'{np.mean(task_res_lst) * 100:.2f}')
    with open(res_file, 'w') as rw:
        rw.write('\t'.join(res_lst))
