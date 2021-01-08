## LM Finetuning

The LM finetuning code is an adaption to a script from the huggingface/pytorch-transformers repository:
* https://github.com/huggingface/pytorch-transformers/blob/v1.0.0/examples/lm_finetuning/finetune_on_pregenerated.py

Prepare the finetuning corpus, here shown for a test corpus "restaurant_corpus_indo.txt":

    --train_corpus /dataset/restaurant_corpus_indo.txt \
    --bert_model bert-base-cased \
    --output_dir review_corpus_prepared \
    --epochs_to_generate 5 --max_seq_len 256

Run actual finetuning with:

    --pregenerated_data review_corpus_prepared \
    --bert_model bert-base-cased \
    --output_dir review_corpus_finetuned \
    --epochs 3 --train_batch_size 8 \

