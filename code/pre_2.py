#coding:utf-8

import numpy as np
import pandas as pd


import sys
stdout = sys.stdout
reload(sys)
sys.setdefaultencoding('utf-8')
sys.stdout = stdout

#全角半角转换
import unicodedata
import re

import warnings
warnings.filterwarnings('ignore')


# In[20]:

# train = pd.read_csv('data/tmp/n_train.csv', sep=',', encoding='utf-8', low_memory=False)
# test = pd.read_csv('data/tmp/n_test.csv',  sep=',',encoding='utf-8', low_memory=False)
train = pd.read_csv('../data/n_train_all_b.csv', sep='$', encoding='utf-8', low_memory=False)
test = pd.read_csv('../data/n_test_all_b.csv',  sep='$',encoding='utf-8', low_memory=False)


# In[21]:

test_ID = test.vid

# o_test=pd.read_csv('data/meinian_round1_test_a_20180409.csv',sep=',',encoding='gbk')
# test_ID = test.vid
# o_test.vid
# all(o_test.vid ==  test.vid)

train_Y = train.iloc[:,1:6]
train_X = train.iloc[:,6:]

test_X = test.iloc[:,6:]
# all(train_X.columns == test_X.columns)

train_Y[u'收缩压'] = train_Y[u'收缩压'].astype(float)
train_Y[u'舒张压'] = train_Y[u'舒张压'].astype(float)
train_Y[u'血清甘油三酯'] = train_Y[u'血清甘油三酯'].astype(float)
train_Y[u'血清高密度脂蛋白'] = train_Y[u'血清高密度脂蛋白'].astype(float)
train_Y[u'血清低密度脂蛋白'] = train_Y[u'血清低密度脂蛋白'].astype(float)


# In[22]:

#
#弃查 = None
#详见报告
#60--70次/分
#10%



def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False
    
def full2half_width(ustr):
    # print type(ustr)  >>> "unicode"
    half = ''
    for u in ustr:
        num = ord(u)
        if num == 0x3000:    # 全角空格变半角
            num = 32
            print 'Found something.'
        elif 0xFF01 <= num <= 0xFF5E:
            num -= 0xfee0
        u = unichr(num)    # to unicode
        half += u
    return half

def trip_dot(x):
    while '.' in  x and  x.index('.') == 0:
        x = x[1:]
    while '.' in  x and x.rindex('.') == len(x)-1:
        x = x[:-1]
    return x     

def check_convert_float(x):
    if x==None:
        return None
    x = unicodedata.normalize('NFKC', x.decode('utf-8'))
    if '|' in  x:
        t = x.split('|')
        t = [trip_dot(s.strip()) for s in t if s not in ['', u'nan', None]]
        if len(t) == 0:
            return None
        if all([isfloat(t[i]) and t[i]==t[0] for i in range(len(t))]):
            return sum([float(i) for i in t]) / len(t)
        else:
            raise ValueError(str(t)+"no float")
    else:
        if isfloat(x):
            return float(x)
        else:
            raise ValueError(str(x)+"no float")       

def check_convert_float_e(x):
    if x==None:
        return None
    #全角转半角
    x = unicodedata.normalize('NFKC', x.strip().decode('utf-8'))
    if '|' in  x:
        t = x.split('|')
        t = [trip_dot(s.strip()) for s in t if s not in ['', u'nan', None]]
        if len(t) == 0:
            return None
        if all([isfloat(t[i]) for i in range(len(t))]):
            return sum([float(i) for i in t]) / len(t)
        else:
            print str(t)+"no float"
#             raise ValueError(str(t)+"no float")
            return None
            
    else:
        if isfloat(x):
            return float(x)
        else:
            print str(x)+"no float"
#             raise ValueError(str(x)+"no float")
            return None
#             
            
def hard_convert_float(x):
    if x==None:
        return None
    #全角转半角
    x = unicodedata.normalize('NFKC', x.strip().decode('utf-8'))
    if '|' in  x:
        t = x.split('|')
        t = [trip_dot(s.strip()) for s in t if s not in ['', u'nan', None]]
        if len(t) == 0:
            return None
        if all([isfloat(t[i]) for i in range(len(t))]):
            return sum([float(i) for i in t]) / len(t)
        else:
            return None
    else:
        if isfloat(x):
            return float(x)
        else:
            return None

def hard_str_float(s):
    s = re.sub(ur'(^阴性$)|(^未见$)|(^正常$)', '0', s)    
    s = re.sub(ur'(^\+$)|(^阳性$)', '999', s)
    p = r'\d+[.。]*\d*[%%]*'
    rep = re.search(p, s)
    if rep == None:
        return None
    else:
#         print(re.search(p, s).group(0))  # 在起始位置匹配
        new_s = rep.group(0)
        x = new_s.replace('。', '.')
        if len(x.split('.'))>2:#2.2.8
            x = '.'.join([i for i in x.split('.') if i != ''])
        if '%' in x:
            x = float(x[:-1]) * 0.01
        return x


def hard_convert_float_fina(x):
    if x==None:
        return None
    #全角转半角
#     print x
    x = unicodedata.normalize('NFKC', x.strip().decode('utf-8'))
    if '|' in  x:
        t = x.split('|')
        t = [hard_str_float(s.strip()) for s in t if s not in ['弃查', '', u'nan', None]]
        t = [s for s in t if s not in [None]]
        if len(t) == 0:
            return None
        if all([isfloat(t[i]) for i in range(len(t))]):
            return sum([float(i) for i in t]) / len(t)
        else:
            return None
    else:
#         print 'assa', x, hard_str_float(x)
        x = hard_str_float(x)
#         print 'asd', x
        if x != None and isfloat(x):
            return float(x)
        else:
            return None


# In[23]:

# test_X.iloc[:, 0] = test_X.iloc[:, 0].astype(str)
# dig_col = []
# e_dig_col = []
# n_dig_col = []


# sub_n = 10000
# per_r = 0.9
# the_df = train_X.iloc[:sub_n, :]
# for c in the_df.columns:
#     try:
#         cnt = the_df[c].astype(str).apply(check_convert_float).isnull().sum()
#         dig_col.append([c, cnt*1.0/sub_n])
#     except:
#         continue
        
# for c in the_df.columns:
#     try:
#         cnt = the_df[c].astype(str).apply(check_convert_float_e).isnull().sum()
#         e_dig_col.append([c, cnt*1.0/sub_n])
#     except:
#         n_dig_col.append(c)
#         continue
# sub_c = [c for c in e_dig_col if c not in dig_col]


dig_col = []
# e_dig_col = []
n_dig_col = []
all_col = []
all_hit_col = []

sub_n = len(train_X)
per_r = 0.8
the_df = train_X.iloc[:sub_n, :]

for c in the_df.columns:
    cnt = sub_n - the_df[c].astype(str).apply(hard_convert_float).isnull().sum()
    n_base = sub_n - the_df[c].isnull().sum()
    all_col.append([c, cnt*1.0/n_base])
    if cnt*1.0/n_base == 1.0:
        all_hit_col.append([c, cnt*1.0/n_base])
    if cnt*1.0/n_base > per_r:
        dig_col.append([c, cnt*1.0/n_base])
    else:
        n_dig_col.append([c, cnt*1.0/n_base])
    # print c, cnt*1.0/n_base


# In[24]:

# 可转换为数字的特征 及其中特征可转换为数字的比例
dig_col.sort(reverse=False, key=lambda x: x[1])
# dig_col[len(dig_col)-1032-20:len(dig_col)-1032]


# In[25]:

# 不可转换为数字的特征 及其中特征可转换为数字的比例
n_dig_col.sort(reverse=True, key=lambda x: x[1])
n_dig_col[:10]
# [i[1] for i in n_dig_col].sort


# In[26]:

# n_dig_col[:3]

new_d_col = dig_col + n_dig_col[:1]
new_d_col.sort(reverse=False, key=lambda x: x[1])
new_d_col[:20]


# In[27]:

# dig_c = [i[0] for i in dig_col]
dig_c = [i[0] for i in new_d_col]

# dig_c


# In[28]:

train_X_d = train_X[dig_c]
for c in dig_c:
    train_X_d[c] = train_X_d[c].astype(str).apply(hard_convert_float_fina)
    # print c, (len(train_X) - train_X_d[c].isnull().sum()) * 1.0 / (len(train_X) - train_X[c].isnull().sum())


# In[29]:

test_X_d = test_X[dig_c]
for c in dig_c:
    test_X_d[c] = test_X_d[c].astype(str).apply(hard_convert_float_fina)
    # print c, (len(test_X) - test_X_d[c].isnull().sum()) * 1.0 / (len(test_X) - test_X[c].isnull().sum())


# In[30]:

test_col_cnt = test_X_d.isnull().sum(axis=0).apply(lambda x: x*1.0/len(test_X_d)).sort_values()


# In[31]:

no_all_null_col = test_col_cnt[test_col_cnt != 1.0].index


# In[32]:

# c = u'2376'
# train_X[c][:50]
# train_X[c].astype(str).apply(check_convert_float_e)
# no_all_null_col
train_X_d_x  = train_X_d[no_all_null_col]
test_X_d_x = test_X_d[no_all_null_col]


# In[33]:

train_X_d_x.insert(loc=0, column='vid', value=train.iloc[:,0])
test_X_d_x.insert(loc=0, column='vid', value=test.iloc[:,0])


# In[34]:

train_X_d_x.to_csv('../data/train_X_d_all_hard_b.csv',index=False, sep=',',encoding='utf-8')

test_X_d_x.to_csv('../data/test_X_d_all_hard_b.csv',index=False, sep=',',encoding='utf-8')


# In[35]:

train.iloc[:,0:6].to_csv('../data/train_Y_d_b.csv',index=False, sep=',',encoding='utf-8')
