from tqdm import tqdm
import spacy
import argparse

parser = argparse.ArgumentParser(description='Generate finetuning corpus for restaurants.')

parser.add_argument('--large',
                    action='store_true',
                    help='export large corpus (10 mio), default is 1 mio')
args = parser.parse_args()

max_sentences = int(10e5)
review_limit = int(150000)

if args.large:  # for 10 Mio Corpus
    review_limit = int(2500000)
    max_sentences = int(15e6)

nlp = spacy.load('en_core_web_sm')
nlp.add_pipe(nlp.create_pipe('sentencizer'))

fn = '/home/muzamil/Dataset/food/Text/MyData/test_data/reviews_indo.txt'
reviews = []

# 4 million reviews to generate about minimum 10 mio sentences
with open(fn) as data_file:
    counter = 0
    for line in data_file:
        counter += 1
        reviews.append(line)
        # if counter == review_limit:
        #     break


# get sentence segmented review with #sentences > 2
def sentence_segment_filter_docs(doc_array):
    sentences = []

    print(str())
    doc_counter = 0
    for doc in nlp.pipe(doc_array, disable=['parser', 'tagger', 'ner'], batch_size=1000, n_threads=8):
        sentences.append([sent.text.strip() for sent in doc.sents])
        doc_counter = doc_counter + 1
        print(doc_counter)
    return sentences


print(f'Found {len(reviews)} restaurant reviews')
print(f'Tokenizing Restaurant Reviews...')

sentences = sentence_segment_filter_docs(reviews)
nr_sents = sum([len(s) for s in sentences])
print(f'Segmented {nr_sents} restaurant sentences')

# Save to file
fn_out = f'/home/muzamil/Dataset/food/Text/MyData/test_data/restaurant_corpus_indo.txt'
with open(fn_out, "w") as f:
    sent_count = 0
    for sents in tqdm(sentences):
        real_sents = []
        for s in sents:
            x = s.replace(' ', '').replace('\n', '').replace('\u200d', '').replace('\u200b', '')
            if x != '':
                real_sents.append(s.replace('\n', '').replace('\u200d', '').replace('\u200b', ''))
        if len(real_sents) >= 2:
            sent_count += len(real_sents)
            str_to_write = "\n" + "\n".join(real_sents) + "\n"
            f.write(str_to_write)

        # if sent_count >= max_sentences:
        #     break

print(f'Done writing to {fn_out}')