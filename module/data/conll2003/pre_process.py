import os
import argparse

import codecs
import pandas as pd

from module import tqdm


def conll2003_preprocess(
        data_dir, train_name="eng.train", dev_name="eng.testa", test_name="eng.testb"):
    train_f = read_data(os.path.join(data_dir, train_name))
    dev_f = read_data(os.path.join(data_dir, dev_name))
    test_f = read_data(os.path.join(data_dir, test_name))

    # create train csv file
    # make a df of labels and text(tokens) column where all the labels from index[0] and tokens from index[1]
    train = pd.DataFrame({"labels": [x[0] for x in train_f], "text": [x[1] for x in train_f]})
    # make another column of cls to check weather the sentence contains any label or not.
    train["cls"] = train["labels"].apply(lambda x: all([y.split("_")[0] == "O" for y in x.split()]))
    train.to_csv(os.path.join(data_dir, "{}.train.csv".format(train_name)), index=False, sep="\t")

    sent_length = []
    sentences = train["text"]
    for item in sentences:
        sent_length.append(len(item.split()))

    print("Max length of sentence {}".format(max(sent_length)))

    # create dev csv file
    dev = pd.DataFrame({"labels": [x[0] for x in dev_f], "text": [x[1] for x in dev_f]})
    dev["cls"] = dev["labels"].apply(lambda x: all([y.split("_")[0] == "O" for y in x.split()]))
    dev.to_csv(os.path.join(data_dir, "{}.dev.csv".format(dev_name)), index=False, sep="\t")

    # create test csv file
    test_ = pd.DataFrame({"labels": [x[0] for x in test_f], "text": [x[1] for x in test_f]})
    test_["cls"] = test_["labels"].apply(lambda x: all([y.split("_")[0] == "O" for y in x.split()]))
    test_.to_csv(os.path.join(data_dir, "{}.dev.csv".format(test_name)), index=False, sep="\t")


def read_data(input_file):
    """Reads a BIO data."""
    with codecs.open(input_file, "r", encoding="utf-8") as f:
        lines = []
        words = []
        labels = []
        f_lines = f.readlines()
        for line in tqdm(f_lines, total=len(f_lines), desc="Process {}".format(input_file)):
            contents = line.strip()
            word = line.strip().split(' ')[0]
            label = line.strip().split(' ')[-1]
            if contents.startswith("-DOCSTART-"):
                words.append('')
                continue

            if len(contents) == 0 and not len(words):
                words.append("")

            # Flattening the list of words(tokens) and list of labels to a single line
            # separated by a space (" ")
            if len(contents) == 0 and words[-1] == '.':
                lbl = ' '.join([label for label in labels if len(label) > 0])
                w = ' '.join([word for word in words if len(word) > 0])
                # list of comma separated labels and tokens
                lines.append([lbl, w])
                words = []
                labels = []
                continue
            words.append(word)
            # replacing the format of label from B-Menu to B_Menu
            labels.append(label.replace("-", "_"))
        return lines


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--train_name', type=str, default="eng.train")
    parser.add_argument('--dev_name', type=str, default="eng.testa")
    parser.add_argument('--test_name', type=str, default="eng.testb")
    return vars(parser.parse_args())


if __name__ == "__main__":
    conll2003_preprocess(**parse_args())