import torch
from torch.utils.data import Dataset
from typing import List, Tuple
import itertools
import os
import json
from tqdm import tqdm
import common.embeddings
import torchtext


class SNLIDataset(Dataset):

    # https://nlp.stanford.edu/projects/snli/

    # url = 'https://nlp.stanford.edu/projects/snli/snli_1.0.zip'

    url = 'https://github.com/danieleds/text-classif-by-untrained-sentence-embeddings/releases/download/0.1/snli.pth'

    embeddings = None

    def __init__(self, examples, root='.data'):
        super(SNLIDataset, self).__init__()
        self.examples = examples
        self.root = root

        if self.__class__.embeddings is None:
            self.__class__.embeddings = common.embeddings.load_fasttext_embeddings(root=root)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        e = self.examples[idx]

        # Apply embeddings
        x1 = common.embeddings.get_labels(e['sentence1'], self.embeddings)
        x2 = common.embeddings.get_labels(e['sentence2'], self.embeddings)

        y = (0 if e['label'] == 'contradiction' else
             1 if e['label'] == 'neutral' else
             2 if e['label'] == 'entailment' else
             -1 if e['label'] == 'hidden' else
             None)

        return {
            'x1': x1,
            'x2': x2,
            'y': torch.as_tensor([y]),
            'id': e['pairId']
        }

    @classmethod
    def load(cls, root='.data'):
        ds_path = os.path.join(root, 'SNLI', 'snli.pth')
        if not os.path.isfile(ds_path):
            if not os.path.exists(os.path.dirname(ds_path)):
                os.makedirs(os.path.dirname(ds_path))
            print('downloading dataset')
            torchtext.utils.download_from_url(SNLIDataset.url, ds_path)
        return torch.load(ds_path)

    @classmethod
    def splits(cls, root='.data'):
        """
        Returns a training set, a validation set, a test set.
        :return: (training set, validation set, test set)
        """

        ds = cls.load(root=root)

        train_fold = ds['folds']['train']
        val_fold = ds['folds']['validation']
        test_fold = ds['folds']['test']

        return cls(train_fold, root=root), cls(val_fold, root=root), cls(test_fold, root=root)

    @classmethod
    def merge_folds(cls, folds: List['SNLIDataset']):
        examples = list(itertools.chain.from_iterable([ f.examples for f in folds ]))
        return cls(examples)


def build_ds(root='.data'):
    # Assume data is downloaded and extracted into source_dir
    source_dir = os.path.join(root, 'SNLI')
    dest_filename = os.path.join(source_dir, 'snli.pth')

    def tokenize_bin_tree(tree: str):
        # Parentheses are "-LRB-" and "-RRB-"
        tokens = [ tkn for tkn in tree.split() if tkn != '(' and tkn != ')']
        tokens = [ '(' if tkn == '-LRB-' else ')' if tkn == '-RRB-' else tkn for tkn in tokens ]
        return tokens

    def parse_line(line: str):
        # Take a JSON line, parse it, tokenize it.
        d = json.loads(line)
        return {
            'label': d['gold_label'],
            'pairId': d['pairID'],
            'sentence1': tokenize_bin_tree(d['sentence1_binary_parse']),
            'sentence2': tokenize_bin_tree(d['sentence2_binary_parse'])
        }

    def parse_file(filename: str):
        examples = [parse_line(line.rstrip('\n')) for line in tqdm(open(filename))]
        # Exclude those examples whose labels are controversial (see dataset paper)
        return [e for e in examples if e['label'] != '-']

    training_set = os.path.join(source_dir, 'snli_1.0', 'snli_1.0_train.jsonl')
    validation_set = os.path.join(source_dir, 'snli_1.0', 'snli_1.0_dev.jsonl')
    test_set = os.path.join(source_dir, 'snli_1.0', 'snli_1.0_test.jsonl')

    ds = {
        'folds': {
            'train': parse_file(training_set),
            'validation': parse_file(validation_set),
            'test': parse_file(test_set),
        }
    }

    # Write the dataset to disk
    torch.save(ds, dest_filename)

    return ds
