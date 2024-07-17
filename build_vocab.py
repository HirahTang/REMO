"""
The vocabulary building scripts.
"""
import os
# import sys

# sys.path.append(sys.path[0]+'/..')

from utils.torchvocab import MolVocab
from data_analysis.USPTO_CONFIG import USPTO_CONFIG

def build():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default=USPTO_CONFIG.full_smiles, type=str)
    parser.add_argument('--vocab_save_folder', default=USPTO_CONFIG.vocab_save_folder, type=str)
    parser.add_argument('--dataset_name', type=str, default=USPTO_CONFIG.dataset_name,
                        help="Will be the first part of the vocab file name. If it is None,"
                             "the vocab files will be: atom_vocab.pkl and bond_vocab.pkl")
    parser.add_argument('--vocab_max_size', type=int, default=None)
    parser.add_argument('--vocab_min_freq', type=int, default=1)
    args = parser.parse_args()

    # fin = open(args.data_path, 'r')
    # lines = fin.readlines()
    for vocab_type in ['atom']:
    # for vocab_type in ['atom', 'bond']:
        vocab_file = f"{vocab_type}_vocab.pkl"
        if args.dataset_name is not None:
            vocab_file = args.dataset_name + '_' + vocab_file
        vocab_save_path = os.path.join(args.vocab_save_folder, vocab_file)

        os.makedirs(os.path.dirname(vocab_save_path), exist_ok=True)
        vocab = MolVocab(file_path=args.data_path,
                         max_size=args.vocab_max_size,
                         min_freq=args.vocab_min_freq,
                         vocab_type=vocab_type)
        print(f"{vocab_type} vocab size", len(vocab))
        vocab.save_vocab(vocab_save_path)


if __name__ == '__main__':
    build()
