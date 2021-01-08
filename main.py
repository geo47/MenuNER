from sklearn_crfsuite.metrics import flat_classification_report

from module.analyze.utils import bert_labels2tokens
from module.data.bert_data import get_data_loader_for_predict
from module.data.conll2003.pre_process import conll2003_preprocess
from module.data import bert_data
from module.models.bert_models import BERTBiLSTMCRF
from module.train.train import NerLearner

from seqeval.metrics import classification_report

from module.utils import print_model_params

import pandas as pd

import warnings
warnings.filterwarnings("ignore")

data_dir = "/home/muzamil/Dataset/food/Text/MyData/CoNLL_2003_dataset/project_data/"


def read_data():
    data = bert_data.LearnData.create(
        train_df_path=data_dir + "eng.train.train.csv",
        valid_df_path=data_dir + "eng.testa.dev.csv",
        idx2labels_path=data_dir + "idx2labels2.txt",
        markup="BIO",
        clear_cache=True,
        model_name="bert-base-cased"
    )
    return data


if __name__ == '__main__':

    conll2003_preprocess(data_dir)

    data = read_data()
    print(data.train_dl.dataset.df)

    device = "cuda:0"

    model = BERTBiLSTMCRF.create(
        len(data.train_ds.idx2label), model_name="/home/muzamil/Projects/Python/NLP/MenuNER/model/FoodieBERT/cased_L-12_H-768_A-12",
        lstm_dropout=0.1, crf_dropout=0.3)

    # print_model_params(model, True, True)

    num_epochs = 10

    learner = NerLearner(
        model, data, data_dir + "conll2003-BERTBiLSTMCRF-base-IO.cpt", t_total=num_epochs * len(data.train_dl))

    print(model.get_n_trainable_params())

    learner.fit(epochs=num_epochs)

    dl = get_data_loader_for_predict(data, df_path=data.valid_ds.config["df_path"])
    print(dl.dataset.df)

    preds = learner.predict(dl)

    pred_tokens, pred_labels = bert_labels2tokens(dl, preds)
    true_tokens, true_labels = bert_labels2tokens(dl, [x.bert_labels for x in dl.dataset])

    assert pred_tokens == true_tokens
    tokens_report = flat_classification_report(true_labels, pred_labels, labels=data.train_ds.idx2label[4:], digits=4)
    print(tokens_report)

    # Test
    dl = get_data_loader_for_predict(data, df_path=data_dir + "eng.testb.dev.csv")
    preds = learner.predict(dl)

    pred_tokens, pred_labels = bert_labels2tokens(dl, preds)
    true_tokens, true_labels = bert_labels2tokens(dl, [x.bert_labels for x in dl.dataset])

    assert pred_tokens == true_tokens
    tokens_report = flat_classification_report(true_labels, pred_labels, labels=data.train_ds.idx2label[4:], digits=4)
    print(tokens_report)

    for true_label in true_labels:
        for l in range(len(true_label)):
            if true_label[l].startswith('B') or true_label[l].startswith('I'):
                if true_label[l] == 'B_O' or true_label[l] == 'I_O':
                    true_label[l] = 'O'
                else:
                    true_label[l] = true_label[l].replace('_', '-')

    for pred_label in pred_labels:
        for l in range(len(pred_label)):
            if pred_label[l].startswith('B') or pred_label[l].startswith('I') or pred_label[l].startswith('[PAD]'):
                if pred_label[l] == 'B_O' or pred_label[l] == 'I_O' or pred_label[l].startswith('[PAD]'):
                    pred_label[l] = 'O'
                else:
                    pred_label[l] = pred_label[l].replace('_', '-')

    t_tokens = []
    t_labels = []
    p_tokens = []
    p_labels = []

    for i in range(len(pred_tokens)):
        for j in range(len(pred_tokens[i])):
            t_tokens.append(true_tokens[i][j])
            t_labels.append(true_labels[i][j])
            p_tokens.append(pred_tokens[i][j])
            p_labels.append(pred_labels[i][j])
            print("[ " + pred_tokens[i][j] + " : " + pred_labels[i][j])
        print("\n")

    # for i in range(len(pred_tokens)):
    #     for j in range(len(pred_tokens[i])):
    #         print("[ " + pred_tokens[i][j] + " : " + pred_labels[i][j])
    #     print("\n")

    report = classification_report(true_labels, pred_labels, digits=4)
    print("\n%s", report)

    # dictionary of lists
    dict = {'true_tokens': t_tokens, 'pred_tokens': p_tokens, 't_labels': t_labels, 'p_labels': p_labels}

    df = pd.DataFrame(dict)

    # saving the dataframe
    df.to_csv('inference.csv')
