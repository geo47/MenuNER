# MenuNER

Implementation for the paper <b>"MenuNER: Food Menu Named Entity Recognition using Domain Adapted Embedding as Feature and Deep Learning Approach"</b>

The best results achieved by the model:

                    precision    recall    f1-score    support

            MENU     0.9386      0.9301    0.9344      1431

       micro avg     0.9386      0.9301    0.9344      1431
       macro avg     0.9386      0.9301    0.9344      1431
       
<h5>Fine-tuning NER task</h5>
        
    ner/preprocess.py 
    --data_dir=../data/menu 
    --bert_model_name_or_path=../model/FoodieBERT/cased_L-12_H-768_A-12 
    --bert_use_sub_label
    
    ner/train.py 
    --data_dir=../data/menu 
    --bert_model_name_or_path=../model/FoodieBERT/cased_L-12_H-768_A-12
    --batch_size=32 
    --eval_batch_size=64 
    --save_path=pytorch-domain-model-bert-en.pt
    --bert_output_dir=bert-domain-checkpoint-en
    --epoch=30
    --bert_use_pos
    --use_char_cnn
    --use_mha
    --bert_use_feature_based
    --use_crf
    
    ner/evaluation.py 
    --config=../configs/config-bert.json
    --data_dir=../data/menu
    --model_path=pytorch-domain-model-bert-en.pt
    --bert_output_dir=bert-domain-checkpoint-en
    --bert_use_pos
    --use_char_cnn
    --use_mha
    --bert_use_feature_based
    --use_crf