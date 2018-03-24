import re
import unicodedata
import numpy as np

from gensim import utils
from gensim.corpora.wikicorpus import WikiCorpus
from tqdm import tqdm

def myTokenizer(content, token_min_len=2, token_max_len=15, lower=True):
    raw = content.split(' ')
    remover = re.compile("[^a-zA-Z-]")
    
    token = []
    
    for i in raw:
        term = remover.sub('', i)
        if lower == True:
            term = term.lower()
        term = unicodedata.normalize('NFKD', term).encode('ascii', 'ignore')
        token.append(term)
    tokenized = filter(None, token)
    
    return [
        utils.to_unicode(token) for token in tokenized
        if token_min_len <= len(token) <= token_max_len and not token.startswith('_')
    ]

file_path = "data/words-corpus/"
input_file = file_path + "idwiki-latest-pages-articles.xml.bz2"
output_file = "corpus.text"
SPACE = ' '

output = open(output_file, 'w')
wiki = WikiCorpus(input_file, lemmatize=False, dictionary={}, tokenizer_func=myTokenizer)

corpus = wiki.get_texts()
# corpus_len = sum(1 for i in corpus) --> 314830

pbar = tqdm(total=314830)
for text in corpus:
    dump = SPACE.join(text) + SPACE
    output.write(dump)
    pbar.update(1)
    
pbar.close()
output.close()