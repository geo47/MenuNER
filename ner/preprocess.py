from __future__ import absolute_import, division, print_function

import sys
import os
import argparse
import random
import logging

from tqdm import tqdm
from ner.util import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_TRAIN_FILE = 'train.txt'
_VALID_FILE = 'valid.txt'
_TEST_FILE  = 'test.txt'
_SUFFIX = '.ids'
_VOCAB_FILE = 'vocab.txt'
_EMBED_FILE = 'embedding.npy'
_POS_FILE = 'pos.txt'
_CHUNK_FILE = 'chunk.txt'
_CHAR_FILE = 'char.txt'
_LABEL_FILE = 'label.txt'
_FSUFFIX = '.fs'


def build_dict(input_path, config):
    logger.info("\n[building dict]")
    poss = {}
    chunks = {}
    chars = {}
    labels = {}
    # add pad, unk info
    poss[config['pad_pos']] = config['pad_pos_id']
    pos_id = 1
    chunks[config['pad_chunk']] = config['pad_chunk_id']
    chunk_id = 1
    chars[config['pad_token']] = config['pad_token_id']
    chars[config['unk_token']] = config['unk_token_id']
    char_id = 2
    labels[config['pad_label']] = config['pad_label_id']
    label_id = 1
    tot_num_line = sum(1 for _ in open(input_path, 'r')) 
    with open(input_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(tqdm(f, total=tot_num_line)):
            line = line.strip()
            if line.startswith("-DOCSTART-"):
                continue
            if line == "": continue
            toks = line.split()
            try:
                assert(len(toks) == 4)
            except Exception as e:
                logger.error(str(idx) + '\t' + line + '\t' + str(e))
                sys.exit(1)
            word = toks[0]
            pos = toks[1]
            chunk = toks[2]
            label = toks[-1]
            if pos not in poss:
                poss[pos] = pos_id
                pos_id += 1
            if chunk not in chunks:
                chunks[chunk] = chunk_id
                chunk_id += 1
            for ch in word:
                if ch not in chars:
                    chars[ch] = char_id
                    char_id += 1
            if label not in labels:
                labels[label] = label_id
                label_id += 1
    logger.info("\nUnique poss, chunk, chars, labels : {}, {}, {}, {}".format(len(poss), len(chunks), len(chars), len(labels)))
    return poss, chunks, chars, labels


def write_dict(dic, output_path):
    logger.info("\n[Writing dict]")
    f_write = open(output_path, 'w', encoding='utf-8')
    for idx, item in enumerate(tqdm(dic.items())):
        _key = item[0]
        _id = item[1]
        f_write.write(_key + ' ' + str(_id))
        f_write.write('\n')
    f_write.close()


# ---------------------------------------------------------------------------- #
# BERT
# ---------------------------------------------------------------------------- #

def build_features(input_path, tokenizer, poss, chunks, labels, config, mode='train'):
    from ner.util_bert import read_examples_from_file
    from ner.util_bert import convert_examples_to_features

    logger.info("[Creating features from file] %s", input_path)
    examples = read_examples_from_file(input_path, mode=mode)
    features = convert_examples_to_features(config, examples, poss, chunks, labels, config['n_ctx'], tokenizer,
                                            cls_token=tokenizer.cls_token,
                                            cls_token_segment_id=0,
                                            sep_token=tokenizer.sep_token,
                                            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                            pad_token_pos_id=config['pad_pos_id'],
                                            pad_token_chunk_id=config['pad_chunk_id'],
                                            pad_token_label_id=config['pad_label_id'],
                                            pad_token_segment_id=0,
                                            sequence_a_segment_id=0)
    return features


def write_features(features, output_path):
    import torch

    logger.info("[Saving features into file] %s", output_path)
    torch.save(features, output_path)


def preprocess_bert(config):
    opt = config['opt']

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(opt.bert_model_name_or_path)

    # build poss, chars, labels
    path = os.path.join(opt.data_dir, _TRAIN_FILE)
    poss, chunks, chars, labels = build_dict(path, config)

    # build features
    path = os.path.join(opt.data_dir, _TRAIN_FILE)
    train_features = build_features(path, tokenizer, poss, chunks, labels, config, mode='train')

    path = os.path.join(opt.data_dir, _VALID_FILE)
    valid_features = build_features(path, tokenizer, poss, chunks, labels, config, mode='valid')

    path = os.path.join(opt.data_dir, _TEST_FILE)
    test_features = build_features(path, tokenizer, poss, chunks, labels, config, mode='test')

    # write features
    path = os.path.join(opt.data_dir, _TRAIN_FILE + _FSUFFIX)
    write_features(train_features, path)

    path = os.path.join(opt.data_dir, _VALID_FILE + _FSUFFIX)
    write_features(valid_features, path)

    path = os.path.join(opt.data_dir, _TEST_FILE + _FSUFFIX)
    write_features(test_features, path)

    # write poss, labels
    path = os.path.join(opt.data_dir, _POS_FILE)
    write_dict(poss, path)
    path = os.path.join(opt.data_dir, _CHUNK_FILE)
    write_dict(chunks, path)
    path = os.path.join(opt.data_dir, _LABEL_FILE)
    write_dict(labels, path)


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--config', type=str, default='../configs/config-bert.json')
    parser.add_argument('--data_dir', type=str, default='../data/conll2003')
    parser.add_argument("--seed", default=5, type=int)
    # for BERT
    parser.add_argument("--bert_model_name_or_path", type=str, default='bert-base-uncased',
                        help="Path to pre-trained model or shortcut name(ex, bert-base-uncased)")
    parser.add_argument('--bert_use_sub_label', action='store_true',
                        help="Set this flag to use sub label instead of using pad label for sub tokens.")
    opt = parser.parse_args()

    # set seed
    random.seed(opt.seed)

    # set config
    config = load_config(opt)
    config['opt'] = opt
    logger.info("%s", config)

    preprocess_bert(config)


if __name__ == '__main__':
    main()
