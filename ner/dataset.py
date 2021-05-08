from __future__ import absolute_import, division, print_function

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def prepare_dataset(config, filepath, DatasetClass, sampling=False, num_workers=1, batch_size=0):
    opt = config['opt']
    dataset = DatasetClass(config, filepath)

    if sampling:
        sampler = RandomSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)
    if hasattr(opt, 'distributed') and opt.distributed:
        sampler = DistributedSampler(dataset)

    bz = opt.batch_size
    if batch_size > 0: bz = batch_size

    loader = DataLoader(dataset, batch_size=bz, num_workers=num_workers, sampler=sampler, pin_memory=True)
    logger.info("[{} data loaded]".format(filepath))
    return loader


class CoNLLBertDataset(Dataset):
    def __init__(self, config, path):
        # load features from file
        features = torch.load(path)
        # convert to tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_pos_ids = torch.tensor([f.pos_ids for f in features], dtype=torch.long)
        all_chunk_ids = torch.tensor([f.chunk_ids for f in features], dtype=torch.long)
        all_char_ids = torch.tensor([f.char_ids for f in features], dtype=torch.long)
        all_word2token_idx = torch.tensor([f.word2token_idx for f in features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)

        self.x = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_pos_ids, all_chunk_ids, all_char_ids,
                               all_word2token_idx)
        self.y = all_label_ids
 
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
