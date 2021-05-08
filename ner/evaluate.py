from __future__ import absolute_import, division, print_function

import os
import argparse
import time
import logging

import torch
import torch.quantization
import numpy as np
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report

from tqdm import tqdm
from ner.util import load_checkpoint, load_config, to_device, to_numpy
from ner.model import BertLSTMCRF
from ner.dataset import prepare_dataset, CoNLLBertDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def set_path(config):
    opt = config['opt']
    opt.data_path = os.path.join(opt.data_dir, 'test.txt.fs')
    opt.label_path = os.path.join(opt.data_dir, 'label.txt')
    opt.pos_path = os.path.join(opt.data_dir, 'pos.txt')
    opt.chunk_path = os.path.join(opt.data_dir, 'chunk.txt')
    opt.test_path = os.path.join(opt.data_dir, 'test.txt')
    opt.vocab_path = os.path.join(opt.data_dir, 'vocab.txt')


def load_model(config, checkpoint):
    opt = config['opt']

    from transformers import AutoTokenizer, AutoConfig, AutoModel
    bert_config = AutoConfig.from_pretrained(opt.bert_output_dir)
    bert_tokenizer = AutoTokenizer.from_pretrained(opt.bert_output_dir)
    bert_model = AutoModel.from_config(bert_config)
    ModelClass = BertLSTMCRF
    model = ModelClass(config, bert_config, bert_model, bert_tokenizer, opt.label_path, opt.pos_path,
                       opt.chunk_path, use_crf=opt.use_crf, use_crf_slice=opt.bert_use_crf_slice,
                       use_pos=opt.bert_use_pos, use_chunk=opt.bert_use_chunk, use_char_cnn=opt.use_char_cnn,
                       use_mha=opt.use_mha, disable_lstm=opt.bert_disable_lstm,
                       feature_based=opt.bert_use_feature_based)
    model.load_state_dict(checkpoint)
    model = model.to(opt.device)
    logger.info("[Loaded]")
    return model


# ---------------------------------------------------------------------------- #
# Evaluation
# ---------------------------------------------------------------------------- #
def write_prediction(config, model, ys, preds, labels):
    opt = config['opt']
    pad_label_id = config['pad_label_id']
    default_label = config['default_label']

    # load test data
    tot_num_line = sum(1 for _ in open(opt.test_path, 'r')) 
    with open(opt.test_path, 'r', encoding='utf-8') as f:
        data = []
        bucket = []
        for idx, line in enumerate(tqdm(f, total=tot_num_line)):
            line = line.strip()
            if line == "":
                data.append(bucket)
                bucket = []
            else:
                entry = line.split()
                assert(len(entry) == 4)
                bucket.append(entry)
        if len(bucket) != 0:
            data.append(bucket)
    # write prediction
    try:
        pred_path = opt.test_path + '.pred'
        with open(pred_path, 'w', encoding='utf-8') as f:
            for i, bucket in enumerate(data):      # foreach sentence
                if i >= ys.shape[0]:
                    logger.info("[Stop to write predictions] : %s" % (i))
                    break
                # use_subtoken = False
                # ys_idx = 0
                # if config['emb_class'] not in ['glove', 'elmo']:
                use_subtoken = True
                ys_idx = 1 # account '[CLS]'
                if opt.bert_use_crf_slice:
                    use_subtoken = False
                    ys_idx = 0
                for j, entry in enumerate(bucket): # foreach token
                    entry = bucket[j]
                    pred_label = default_label
                    if ys_idx < ys.shape[1]:
                        pred_label = labels[preds[i][ys_idx]]
                    entry.append(pred_label)
                    f.write(' '.join(entry) + '\n')
                    if use_subtoken:
                        word = entry[0]
                        word_tokens = model.bert_tokenizer.tokenize(word)
                        ys_idx += len(word_tokens)
                    else:
                        ys_idx += 1
                f.write('\n')
    except Exception as e:
        logger.warn(str(e))


def prepare_datasets(config):
    opt = config['opt']
    DatasetClass = CoNLLBertDataset
    test_loader = prepare_dataset(config, opt.data_path, DatasetClass, sampling=False, num_workers=1)
    return test_loader


def evaluate(opt):
    # set config
    config = load_config(opt)
    if opt.num_threads > 0: torch.set_num_threads(opt.num_threads)
    config['opt'] = opt
    logger.info("%s", config)

    # set path
    set_path(config)

    # prepare test dataset
    test_loader = prepare_datasets(config)
 
    # load pytorch model checkpoint
    checkpoint = load_checkpoint(opt.model_path, device=opt.device)

    # prepare model and load parameters
    model = load_model(config, checkpoint)
    model.eval()

    # evaluation
    preds = None
    ys    = None
    n_batches = len(test_loader)
    total_examples = 0
    whole_st_time = time.time()
    first_time = time.time()
    first_examples = 0
    total_duration_time = 0.0
    with torch.no_grad():
        for i, (x,y) in enumerate(tqdm(test_loader, total=n_batches)):
            start_time = time.time()
            x = to_device(x, opt.device)
            y = to_device(y, opt.device)
            if opt.use_crf and opt.bert_use_crf_slice:
                # slice y to remain first token's of word's
                word2token_idx = x[6]
                mask = torch.sign(torch.abs(word2token_idx)).to(torch.uint8).to(opt.device)
                y = y.gather(1, word2token_idx)
                y *= mask
            if opt.use_crf:
                logits, prediction = model(x)
            else:
                logits = model(x)
                logits = torch.softmax(logits, dim=-1)

            if preds is None:
                if opt.use_crf: preds = to_numpy(prediction)
                else: preds = to_numpy(logits)
                ys = to_numpy(y)
            else:
                if opt.use_crf:
                    preds = np.append(preds, to_numpy(prediction), axis=0)
                else:
                    preds = np.append(preds, to_numpy(logits), axis=0)
                ys = np.append(ys, to_numpy(y), axis=0)
            cur_examples = y.size(0)
            total_examples += cur_examples
            if i == 0: # first one may take longer time, so ignore in computing duration.
                first_time = float((time.time()-first_time)*1000)
                first_examples = cur_examples
            if opt.num_examples != 0 and total_examples >= opt.num_examples:
                logger.info("[Stop Evaluation] : up to the {} examples".format(total_examples))
                break
            duration_time = float((time.time()-start_time)*1000)
            if i != 0: total_duration_time += duration_time
            '''
            logger.info("[Elapsed Time] : {}ms".format(duration_time))
            '''
    whole_time = float((time.time()-whole_st_time)*1000)
    avg_time = (whole_time - first_time) / (total_examples - first_examples)
    if not opt.use_crf:
        preds = np.argmax(preds, axis=2)
    # compute measure using seqeval
    labels = model.labels
    ys_lbs = [[] for _ in range(ys.shape[0])]
    preds_lbs = [[] for _ in range(ys.shape[0])]
    pad_label_id = config['pad_label_id']
    for i in range(ys.shape[0]):     # foreach sentence
        for j in range(ys.shape[1]): # foreach token
            if ys[i][j] != pad_label_id:
                ys_lbs[i].append(labels[ys[i][j]])
                preds_lbs[i].append(labels[preds[i][j]])
    ret = {
        "precision": precision_score(ys_lbs, preds_lbs),
        "recall": recall_score(ys_lbs, preds_lbs),
        "f1": f1_score(ys_lbs, preds_lbs),
        "report": classification_report(ys_lbs, preds_lbs, digits=4),
    }
    print(ret['report'])
    f1 = ret['f1']
    # write predicted labels to file
    write_prediction(config, model, ys, preds, labels)

    logger.info("[F1] : {}, {}".format(f1, total_examples))
    logger.info("[Elapsed Time] : {} examples, {}ms, {}ms on average".format(total_examples, whole_time, avg_time))
    logger.info("[Elapsed Time(total_duration_time, average)] : {}ms, {}ms".format(
        total_duration_time, total_duration_time/(total_examples-1)))


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--config', type=str, default='configs/config-bert.json')
    parser.add_argument('--data_dir', type=str, default='data/conll2003')
    parser.add_argument('--model_path', type=str, default='pytorch-model-glove.pt')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_threads', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_examples', default=0, type=int,
                        help="Number of examples to evaluate, 0 means all of them.")
    parser.add_argument('--use_crf', action='store_true', help="Add CRF layer")
    parser.add_argument('--use_char_cnn', action='store_true', help="Add Character features")
    parser.add_argument('--use_mha', action='store_true', help="Add Multi-Head Attention layer.")
    # for BERT
    parser.add_argument('--bert_output_dir', type=str, default='bert-checkpoint',
                        help="The checkpoint directory of fine-tuned BERT model.")
    parser.add_argument('--bert_use_feature_based', action='store_true',
                        help="Use BERT as feature-based, default fine-tuning")
    parser.add_argument('--bert_disable_lstm', action='store_true',
                        help="Disable lstm layer")
    parser.add_argument('--bert_use_pos', action='store_true', help="Add Part-Of-Speech features")
    parser.add_argument('--bert_use_chunk', action='store_true', help="Add Chunk features")
    parser.add_argument('--bert_use_crf_slice', action='store_true',
                        help="Set this flag to slice logits before applying crf layer.")

    opt = parser.parse_args()

    evaluate(opt) 


if __name__ == '__main__':
    main()
