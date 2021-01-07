from module.models.bert_models import BERTBiLSTMCRF
from module.data.bert_data import get_data_loader_for_predict
from module.data import bert_data
from module.train.train import NerLearner
from module.analyze.utils import bert_labels2tokens


idx2label = ['[PAD]', '[CLS]', '[SEP]', 'X', 'O', 'B_MENU', 'I_MENU']

data_dir = "/home/muzamil/Dataset/food/Text/MyData/CoNLL_2003_dataset/project_data/"


def read_data():
    data = bert_data.LearnData.create(
        train_df_path=data_dir + "eng.train.train.csv",
        valid_df_path=data_dir + "infer.csv",
        idx2labels_path=data_dir + "idx2labels2.txt",
        markup="BIO",
        clear_cache=True,
        model_name="bert-base-cased"
    )
    return data

data = read_data()
print(data.train_dl.dataset.df)

model = BERTBiLSTMCRF.create(
        len(data.train_ds.idx2label), model_name="/home/muzamil/Projects/Python/NLP/BERT-NER/model/my_model/cased_L-12_H-768_A-12/",
        lstm_dropout=0.1, crf_dropout=0.3)


dl = get_data_loader_for_predict(data, df_path=data.valid_ds.config["df_path"])
print(dl.dataset.df)

learner = NerLearner(
        model, data, data_dir + "conll2003-BERTBiLSTMCRF-base-IO.cpt")
preds = learner.predict(dl)

pred_tokens, pred_labels = bert_labels2tokens(dl, preds)

assert len(pred_tokens) == len(pred_labels)

print(pred_tokens)
print(pred_labels)

for i in range(len(pred_tokens)):
    for j in range(len(pred_tokens[i])):
        print("[ "+pred_tokens[i][j]+" : "+pred_labels[i][j])
    print("\n")