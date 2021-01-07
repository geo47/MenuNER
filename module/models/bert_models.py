from torch import nn

import abc  # Abstract Base Class

from module.layers.decoders import CRFDecoder
from module.layers.embedders import BERTEmbedder
from module.layers.layers import BiLSTM


class BERTNerModel(nn.Module, metaclass=abc.ABCMeta):
    """Base class for all BERT Models"""

    @abc.abstractmethod
    def forward(self, batch):
        raise NotImplementedError("abstract method forward must be implemented")

    @abc.abstractmethod
    def score(self, batch):
        raise NotImplementedError("abstract method score must be implemented")

    @abc.abstractmethod
    def create(self, *args, **kwargs):
        raise NotImplementedError("abstract method create must be implemented")

    def get_n_trainable_params(self):
        pp = 0
        for p in list(self.parameters()):
            if p.requires_grad:
                num = 1
                for s in list(p.size()):
                    num = num * s
                pp += num
        return pp


class BERTBiLSTMCRF(BERTNerModel):
    ''' BERTBiLSTMCRF class extends BERTNerModel class '''

    def __init__(self, embeddings, lstm, crf, device="cuda"):
        super(BERTBiLSTMCRF, self).__init__()
        self.embeddings = embeddings
        self.lstm = lstm
        self.crf = crf
        self.to(device)

    def forward(self, batch):
        input_, labels_mask, input_type_ids = batch[:3]
        input_embeddings = self.embeddings(batch)
        output, _ = self.lstm.forward(input_embeddings, labels_mask)
        return self.crf.forward(output, labels_mask)

    def score(self, batch):
        input_, labels_mask, input_type_ids, labels = batch
        input_embeddings = self.embeddings(batch)
        output, _ = self.lstm.forward(input_embeddings, labels_mask)
        return self.crf.score(output, labels_mask, labels)

    @classmethod
    def create(cls,
               label_size,
               # BertEmbedder params
               model_name='bert-base-cased', mode="weighted", is_freeze=True,
               # BiLSTM params
               embedding_size=768, hidden_dim=512, rnn_layers=1, lstm_dropout=0.3,
               # CRFDecoder params
               crf_dropout=0.5,
               # Global params
               device="cuda"):

        embeddings = BERTEmbedder.create(model_name=model_name, device=device, mode=mode, is_freeze=is_freeze)
        lstm = BiLSTM.create(
            embedding_size=embedding_size, hidden_dim=hidden_dim, rnn_layers=rnn_layers, dropout=lstm_dropout)
        crf = CRFDecoder.create(label_size, hidden_dim, crf_dropout)
        return cls(embeddings, lstm, crf, device)