
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


data_path = "data/words-corpus/"
filename = "lapor.csv"


# In[3]:


train_df = pd.read_csv(data_path + filename, delimiter=',')
train_df.head()


# In[4]:


kategori_total_list = train_df["Kategori"].value_counts()
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print kategori_total_list


# In[5]:


kategori = train_df["Kategori"].values
for i in range(len(kategori)):
    kat = str(kategori[i]).lower()
    if "sehat" in kat or "obat" in kat or "jamkesmas" in kat:
        kategori[i] = "Kesehatan"
    elif "banjir" in kat or "bencana" in kat:
        kategori[i] = "Lingkungan Hidup dan Penanggulangan Bencana"
    elif "bbm" in kat:
        kategori[i] = "BBM"
    elif "perekonomian" in kat:
        kategori[i] = "Perekonomian"
    elif "usaha" in kat:
        kategori[i] = "Perdagangan, Perindustrian, Iklim Usaha, dan Investasi"
    elif "pendidikan" in kat:
        kategori[i] = "Pendidikan"
    elif "kewaspadaan" in kat or "perundungan" in kat or "kantor cabang" in kat or "sms" in kat or "haji" in kat or "pemberdayaan masyarakat" in kat or "situasi khusus" in kat:
        kategori[i] = "Topik Lainnya"


# In[6]:


train_df["Kategori Edited"] = kategori


# In[7]:


kategori_total_list = train_df["Kategori"].value_counts()
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print kategori_total_list


# In[8]:


laporan = train_df["IsiLaporan"].values
laporan


# In[9]:


def getWordEmbedding(word, cursor):
#     word = word.replace("'", "''")
    sql = """select vec from term where term like %s"""
    cursor.execute(sql, (str(word),))
    data = cursor.fetchall()
    if len(data) > 0:
        decoded_vec = json.JSONDecoder().decode(data[0][0])
        vec = np.asarray(decoded_vec, dtype=np.float32)
        return True, vec
    else:
        return False, data
    
def myTokenizer(content, lower=True):
    raw = content.split(' ')
    remover = re.compile("[^a-zA-Z-]")
    
    token = []
    
    for i in raw:
        term = remover.sub('', i)
        if lower == True:
            term = term.lower()
        token.append(term)
    tokenized = filter(None, token)
    
    return tokenized

def sentenceToVec(string):
    if type(string) is not str:
        return np.zeros((50,), dtype=float)
    
    string = string.replace('\n', '')
    string = np.array(myTokenizer(string))
    
    feature = None
    begin = True
    for word in string:
        stat, vec = getWordEmbedding(word, cursor)
        if not stat:
            continue
        if begin:
            begin = False
            feature = vec
        else:
            feature += vec
            # feature = np.concatenate([feature, vec])
    
    if feature is not None:
        feature = feature/np.linalg.norm(feature)
    else:
        feature = np.zeros((50,), dtype=float)
    
    return feature

import re
import json
import unicodedata
import mysql.connector
from tqdm import tqdm

db = mysql.connector.connect(user="root", password='misfanatik', database="glove")
cursor = db.cursor(buffered=True)

pbar = tqdm(total=len(laporan))
ftr = []
for i in laporan:
    try:
        ftr.append(sentenceToVec(i))
    except Exception as err:
        print err
        print i
        break
    pbar.update(1)
pbar.close()
ftr = np.array(ftr)


# In[12]:


nan_idx = []
for i in range(len(kategori)):
    if kategori[i] is np.nan:
        nan_idx.append(i)
kategori_c = np.delete(kategori, nan_idx)
ftr_c = np.delete(ftr, nan_idx, 0)


# In[13]:


cls = []
kategori_u = pd.unique(kategori_c)
kategori_u = kategori_u.tolist()
for i in range(len(kategori_c)):
    one_hot = np.zeros((len(kategori_u),), dtype=int)
    idx = kategori_u.index(kategori_c[i])
    one_hot[idx] = 1
    cls.append(one_hot)
cls = np.array(cls)
    
print "len of ftr " + str(len(ftr_c))
print "len of cls " + str(len(cls))


# In[14]:


np.save("ftr-data-source.npy", ftr_c)
np.save("cls-data-source.npy", cls)

