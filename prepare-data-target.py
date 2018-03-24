import re
import json
import unicodedata
import mysql.connector
import numpy as np
# import tflearn
# import numpy as np
# import tensorflow as tf
# from tflearn.layers.core import input_data, dropout, fully_connected
# from tflearn.layers.conv import conv_1d, max_pool_1d
# from tflearn.layers.estimator import regression

from gensim import utils

MULAI_LAPORAN = 0
SELESAI_LAPORAN = 1
BATAL_LAPORAN = 2
ABOUT = 3
HELP = 4
MENOLAK = 5

def padSequence(feature, max_feature):
    pad = np.zeros(max_feature-len(feature), dtype=np.float32)
    feature = np.concatenate([feature, pad])
    return feature

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

def arrangeInputMatrix(data_path, cls_index):
    ftr_list = []
    cls_list = []
    
    data = open(data_path)
    cls = np.zeros((6,), dtype=int)
    cls[cls_index] = 1
    
    for i in data:
        string = i.replace('\n', '')
        string = np.array(myTokenizer(string))

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

        feature = feature/np.linalg.norm(feature)
        ftr_list.append(feature)
        cls_list.append(cls)
        
    return np.array(ftr_list), np.array(cls_list)
    
db = mysql.connector.connect(user="root", password='misfanatik', database="glove")
cursor = db.cursor(buffered=True)

data_dir = "data/data-target/"
ftr_1, cls_1 = arrangeInputMatrix(data_dir + "data_mulai_laporan.text", MULAI_LAPORAN)
ftr_2, cls_2 = arrangeInputMatrix(data_dir + "data_batal_laporan.text", BATAL_LAPORAN)
ftr_3, cls_3 = arrangeInputMatrix(data_dir + "data_selesai_laporan.text", SELESAI_LAPORAN)
ftr_4, cls_4 = arrangeInputMatrix(data_dir + "data_about.text", ABOUT)
ftr_5, cls_5 = arrangeInputMatrix(data_dir + "data_help.text", HELP)
ftr_6, cls_6 = arrangeInputMatrix(data_dir + "data_menolak.text", MENOLAK)

ftr_final = np.vstack((ftr_1, ftr_2, ftr_3, ftr_4, ftr_5, ftr_6))
cls_final = np.vstack((cls_1, cls_2, cls_3, cls_4, cls_5, cls_6))

print len(ftr_final)
print len(cls_final)

np.save("ftr-data-target.npy", ftr_final)
np.save("cls-data-target.npy", cls_final)

