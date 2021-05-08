from __future__ import absolute_import, division, print_function

import sys

from allennlp.modules.elmo import batch_to_ids
from tqdm import tqdm

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------- #
# base code from
#     https://github.com/huggingface/transformers/blob/master/examples/utils_ner.py
# ---------------------------------------------------------------------------- #

class InputExample(object):
    def __init__(self, guid, words, poss, chunks, labels):
        self.guid   = guid
        self.words  = words
        self.poss   = poss
        self.chunks = chunks
        self.labels = labels


class InputFeature(object):
    def __init__(self, input_ids, input_mask, segment_ids, pos_ids, chunk_ids, char_ids, label_ids, word2token_idx):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.pos_ids = pos_ids
        self.chunk_ids = chunk_ids
        self.char_ids = char_ids
        self.label_ids = label_ids
        self.word2token_idx = word2token_idx


def read_examples_from_file(file_path, mode='train'):
    guid_index = 1
    examples = []
    tot_num_line = sum(1 for _ in open(file_path, 'r'))
    with open(file_path, encoding="utf-8") as f:
        bucket = []
        for idx, line in enumerate(tqdm(f, total=tot_num_line)):
            line = line.strip()
            if line.startswith("-DOCSTART-"):
                continue
            if line == "":
                tokens = []
                posseq = []
                chunkseq = []
                labelseq = []
                for entry in bucket:
                    token = entry[0]  # first column from CoNLL2003 dataset (Token)
                    pos = entry[1]  # second column from CoNLL2003 dataset (POS)
                    chunk = entry[2]  # third column from CoNLL2003 dataset (Chunk)
                    label = entry[3]  # last column from CoNLL2003 dataset (Label)
                    tokens.append(token)
                    posseq.append(pos)
                    chunkseq.append(chunk)
                    labelseq.append(label)
                examples.append(InputExample(guid="{}-{}".format(mode, guid_index),
                                             words=tokens,
                                             poss=posseq,
                                             chunks=chunkseq,
                                             labels=labelseq))
                guid_index += 1
                bucket = []
            else:
                entry = line.split()
                assert(len(entry) == 4)
                bucket.append(entry)

        # Check for the last entry if it doesn't match the (line == "") condition
        if len(bucket) != 0:
            tokens = []
            posseq = []
            chunkseq = []
            labelseq = []
            for entry in bucket:
                token = entry[0]
                pos = entry[1]
                chunk = entry[2]
                label = entry[3]
                tokens.append(token)
                posseq.append(pos)
                chunkseq.append(chunk)
                labelseq.append(label)
            examples.append(InputExample(guid="{}-{}".format(mode, guid_index),
                                         words=tokens,
                                         poss=posseq,
                                         chunks=chunkseq,
                                         labels=labelseq))
            guid_index += 1

    return examples

def convert_single_example_to_feature(config,
                                      example,
                                      pos_map,
                                      chunk_map,
                                      label_map,
                                      max_seq_length,
                                      tokenizer,
                                      cls_token="[CLS]",
                                      cls_token_segment_id=0,
                                      sep_token="[SEP]",
                                      pad_token=0,
                                      pad_token_pos_id=0,
                                      pad_token_chunk_id=0,
                                      pad_token_label_id=0,
                                      pad_token_segment_id=0,
                                      sequence_a_segment_id=0,
                                      ex_index=-1):
    """
    -> input_ids: The input ids are often the only required parameters to be passed to the model as input.
            They are token indices, numerical representations of tokens building the sequences that will be
            used as input by the model. BERT tokenizer uses is a WordPiece tokenizer.
    -> token_type_ids: Some models’ purpose is to do sequence classification or question answering.
            These require two different sequences to be joined in a single “input_ids” entry,
            which usually is performed with the help of special tokens, such as the classifier ([CLS]) and
            separator ([SEP]) tokens. For example, the BERT model builds its two sequence input as such:
            "[CLS] SEQUENCE_A [SEP] SEQUENCE_B [SEP]" (indicates fist sequence by 0 and second by 1)
    -> attention_mask: The attention mask is an optional argument used when batching sequences together.
            This argument indicates to the model which tokens should be attended to (indicates by 1),
            and which should not (indicates by 0).

    convention in BERT:
    for single sequence:
      word      : the dog is hairy
      word_idx  : 0   1   2  3                                                | params
      ----------------------------------------------------------------------- | -------------- |
      tokens:        [CLS] the dog is ha ##iry . [SEP] <pad> <pad> <pad> ...  |                |
      token_idx:       0   1   2   3  4  5     6   7     8     9     10  ...  |                |
      input_ids:       x   x   x   x  x  x     x   x     0     0     0   ...  | input_ids      |
      segment_ids:     0   0   0   0  0  0     0   0     0     0     0   ...  | token_type_ids |
      input_mask:      1   1   1   1  1  1     1   1     0     0     0   ...  | attention_mask |
      label_ids:       0   1   1   1  1  0     1   0     0     0     0   ...  |                |
      ----------------------------------------------------------------------- |                |
      pos_ids:         0   10  2   ...
      char_ids:        [0,..., 0] [259, ..., 261] ...
      -----------------------------------------------------------------------
      idx              0   1   2   3
      word2token_idx:  1   2   3   4  0  0  0 ...  
      word2token_idx[idx] = token_idx
      -----------------------------------------------------------------------
    """

    opt = config['opt']
    tokens = []
    pos_ids = []
    chunk_ids = []
    char_ids = []
    label_ids = []
    word2token_idx = []
    token_idx = 1  # consider first sub-token is '[CLS]'
    pad_char_ids = [config['pad_token_id']] * config['char_n_ctx']

    for word, pos, chunk, label in zip(example.words, example.poss, example.chunks, example.labels):
        # word extension
        word_tokens = tokenizer.tokenize(word)  # ['Order', '##ed']
        tokens.extend(word_tokens)
        # build word2token_idx, save the first token's idx of sub-tokens for the word.
        # token_idx must be less than max_seq_length.
        if token_idx < max_seq_length:
            word2token_idx.append(token_idx)
            token_idx += len(word_tokens)
        # pos extension: set same pos_id
        pos_id = pos_map[pos]   # get pos ids by pos tag from pos.txt file
        pos_ids.extend([pos_id] + [pos_id] * (len(word_tokens) - 1))    # extends pos_id to word_token level
        # chunk extension: set same chunk_id
        chunk_id = chunk_map[chunk]     # get chunk ids by chunk tag from chunk.txt file
        chunk_ids.extend([chunk_id] + [chunk_id] * (len(word_tokens) - 1))  # extends chunk_id to word_token level
        # char extension
        c_ids = batch_to_ids([word_tokens])[0].detach().cpu().numpy().tolist()  # get char ids
        char_ids.extend(c_ids)  # extends char_ids to word_token level (for ['Order', '##ed'], [[234, 232,..], [234, 232,..]])
        # label extension: set pad_token_label_id
        label_id = label_map[label] # get label ids by label tag from label.txt file
        if opt.bert_use_sub_label:
            if label == config['default_label']:
                # ex) 'round', '##er' -> 1/'O', 1/'O'
                sub_token_label = label
                sub_token_label_id = label_map[sub_token_label]
                label_ids.extend([label_id] + [sub_token_label_id] * (len(word_tokens) - 1))
            else:
                # ex) 'BR', '##US', '##SE', '##LS' -> 6/'B-LOC', 9/'I-LOC', 9/'I-LOC', 9/'I-LOC'
                sub_token_label = label
                prefix, suffix = label.split('-', maxsplit=1)
                if prefix == 'B': sub_token_label = 'I-' + suffix
                sub_token_label_id = label_map[sub_token_label]
                label_ids.extend([label_id] + [sub_token_label_id] * (len(word_tokens) - 1))
        else:
            label_ids.extend([label_id] + [pad_token_label_id] * (len(word_tokens) - 1))  # extends label_id to word_token level

    if len(tokens) != len(pos_ids):
        # tokenizer returns empty result, ex) [<96>, ;, -, O], [<94>, ``, -, O]
        logger.info("guid: %s", example.guid)
        logger.info("words: %s", " ".join([str(x) for x in example.words]))
        logger.info('len(words): ' + str(len(example.words)))
        logger.info("poss: %s", " ".join([str(x) for x in example.poss]))
        logger.info('len(poss): ' + str(len(example.poss)))
        logger.info("chunks: %s", " ".join([str(x) for x in example.chunks]))
        logger.info('len(chunks): ' + str(len(example.chunks)))
        logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
        logger.info('len(tokens): ' + str(len(tokens)))
        logger.info("pos_ids: %s", " ".join([str(x) for x in pos_ids]))
        logger.info("chunk_ids: %s", " ".join([str(x) for x in chunk_ids]))
        logger.info('len(pos_ids): ' + str(len(pos_ids)))
        logger.info("char_ids: %s", " ".join([str(x) for x in char_ids]))
        logger.info('len(char_ids): ' + str(len(char_ids)))
        sys.exit(1)

    special_tokens_count = 2 #if sep_token_extra else 2
    if len(tokens) > max_seq_length - special_tokens_count:
        tokens = tokens[:(max_seq_length - special_tokens_count)]
        pos_ids = pos_ids[:(max_seq_length - special_tokens_count)]
        chunk_ids = chunk_ids[:(max_seq_length - special_tokens_count)]
        char_ids = char_ids[:(max_seq_length - special_tokens_count)]
        label_ids = label_ids[:(max_seq_length - special_tokens_count)]

    tokens += [sep_token]
    pos_ids += [pad_token_pos_id]
    chunk_ids += [pad_token_chunk_id]
    char_ids += [pad_char_ids]
    label_ids += [pad_token_label_id]
    segment_ids = [sequence_a_segment_id] * len(tokens)

    tokens = [cls_token] + tokens
    segment_ids = [cls_token_segment_id] + segment_ids
    pos_ids = [pad_token_pos_id] + pos_ids
    chunk_ids = [pad_token_chunk_id] + chunk_ids
    char_ids = [pad_char_ids] + char_ids
    label_ids = [pad_token_label_id] + label_ids

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)

    # zero-pad up to the sequence length.
    padding_length = max_seq_length - len(input_ids)
    input_ids += ([pad_token] * padding_length)
    input_mask += ([0] * padding_length)
    segment_ids += ([pad_token_segment_id] * padding_length)
    pos_ids += ([pad_token_pos_id] * padding_length)
    chunk_ids += ([pad_token_chunk_id] * padding_length)
    char_ids += ([pad_char_ids] * padding_length)
    label_ids += ([pad_token_label_id] * padding_length)
    padding_length_for_word2token_idx = max_seq_length - len(word2token_idx)
    word2token_idx += ([0] * padding_length_for_word2token_idx)  # padding means the first token embedding will be used as dummy.

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(pos_ids) == max_seq_length
    assert len(chunk_ids) == max_seq_length
    assert len(char_ids) == max_seq_length
    assert len(label_ids) == max_seq_length
    assert len(word2token_idx) == max_seq_length

    if ex_index != -1 and ex_index < 5:
        logger.info("*** Example ***")
        logger.info("guid: %s", example.guid)
        logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
        logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
        logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
        logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
        logger.info("pos_ids: %s", " ".join([str(x) for x in pos_ids]))
        logger.info("chunk_ids: %s", " ".join([str(x) for x in chunk_ids]))
        logger.info("char_ids: %s ...", " ".join([str(x) for x in char_ids][:3]))
        logger.info("label_ids: %s", " ".join([str(x) for x in label_ids]))
        logger.info("word2token_idx: %s", " ".join([str(x) for x in word2token_idx]))

    feature = InputFeature(input_ids=input_ids,
                            input_mask=input_mask,
                            segment_ids=segment_ids,
                            pos_ids=pos_ids,
                            chunk_ids=chunk_ids,
                            char_ids=char_ids,
                            label_ids=label_ids,
                            word2token_idx=word2token_idx)
    return feature


def convert_examples_to_features(config,
                                 examples,
                                 pos_map,
                                 chunk_map,
                                 label_map,
                                 max_seq_length,
                                 tokenizer,
                                 cls_token="[CLS]",
                                 cls_token_segment_id=0,
                                 sep_token="[SEP]",
                                 pad_token=0,
                                 pad_token_pos_id=0,
                                 pad_token_chunk_id=0,
                                 pad_token_label_id=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0):

    features = []
    for (ex_index, example) in enumerate(tqdm(examples)):
        '''
        if ex_index % 1000 == 0:get_examples_dataset
            logger.info("Writing example %d of %d", ex_index, len(examples))
        '''

        feature = convert_single_example_to_feature(config,
                                                    example,
                                                    pos_map,
                                                    chunk_map,
                                                    label_map,
                                                    max_seq_length,
                                                    tokenizer,
                                                    cls_token=cls_token,
                                                    cls_token_segment_id=cls_token_segment_id,
                                                    sep_token=sep_token,
                                                    pad_token=pad_token,
                                                    pad_token_pos_id=pad_token_pos_id,
                                                    pad_token_chunk_id=pad_token_chunk_id,
                                                    pad_token_label_id=pad_token_label_id,
                                                    pad_token_segment_id=pad_token_segment_id,
                                                    sequence_a_segment_id=sequence_a_segment_id,
                                                    ex_index=ex_index)
        features.append(feature)
    return features
