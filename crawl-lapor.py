
# coding: utf-8

# In[1]:


import re
import unicodedata
import pandas as pd

from gensim import utils
from tqdm import tqdm


# In[2]:


file_path = "data/words-corpus/"
data = pd.read_csv(file_path + "lapor.csv", delimiter=',')


# In[3]:


data.head()


# In[4]:


def myTokenizer(content, token_min_len=2, token_max_len=15, lower=True):
    raw = content.split(' ')
    remover = re.compile("[^a-zA-Z-]")
    
    token = []
    
    for i in raw:
        term = remover.sub('', i)
        if lower == True:
            term = term.lower()
        token.append(term)
    tokenized = filter(None, token)
    
    return [
        utils.to_unicode(token) for token in tokenized
        if token_min_len <= len(token) <= token_max_len and not token.startswith('_')
    ]


# In[5]:


output_file = "corpus.text"
output = open(output_file, 'a')
SPACE = ' '

sentences = data["IsiLaporan"].values
pbar = tqdm(total=len(sentences))
for i in sentences:
    try:
        tokenized_sentence = myTokenizer(i)
        dump = SPACE.join(tokenized_sentence) + SPACE
        output.write(dump)
        pbar.update(1)
    except:
        continue
        pbar.update(1)
    
pbar.close()
output.close()

