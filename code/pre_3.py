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


# In[40]:

train_X_o = pd.read_csv('../data/train_X_d_all_hard_b.csv', sep=',', encoding='utf-8', low_memory=False)
# test_X_o = pd.read_csv('data/tmp/test_X_d_all_hard.csv',  sep=',',encoding='utf-8', low_memory=False)
# train_Y_o = pd.read_csv('data/tmp/train_Y_d.csv', sep=',', encoding='utf-8', low_memory=False)

# test_ID = test_X_o.vid
train_X_c = train_X_o.drop('vid', axis=1)
train_X_c = train_X_c.columns
# train_X_c[:10]
# test_X = test_X_o.drop('vid', axis=1)
# train_Y = train_Y_o.drop('vid', axis=1)


# In[41]:

train_X_d_c = train_X_c.values


# In[42]:

train = pd.read_csv('../data/n_train_all_b.csv', sep='$', encoding='utf-8', low_memory=False)
test = pd.read_csv('../data/n_test_all_b.csv',  sep='$',encoding='utf-8', low_memory=False)


# In[43]:

# train = train[~train[u'收缩压'].isnull()]

train_Y = train.iloc[:,1:6]
train_X = train.iloc[:,6:]

test_ID = test.vid
test_X = test.iloc[:,6:]


# In[44]:

train_Y[u'收缩压'] = train_Y[u'收缩压'].astype(float)
train_Y[u'舒张压'] = train_Y[u'舒张压'].astype(float)
train_Y[u'血清甘油三酯'] = train_Y[u'血清甘油三酯'].astype(float)
train_Y[u'血清高密度脂蛋白'] = train_Y[u'血清高密度脂蛋白'].astype(float)
train_Y[u'血清低密度脂蛋白'] = train_Y[u'血清低密度脂蛋白'].astype(float)


# In[45]:

remain_c = list(set(train_X.columns) - set(train_X_d_c))
len(remain_c)
train_X_t = train_X[remain_c]


# In[46]:

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
#     s = re.sub(ur'(^阴性$)|(^未见$)', '0', s)    
#     s = re.sub(ur'(^\+$)|(^阳性$)', '999', s)
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
    
def check_pn(x):
    # 定型
    _p, _n = 0, 0
    
    p = re.compile(r'(阳性|[^\d]\+[^\d]|^正常$|Normal|^未见$|^未检出$)')
    rep = p.findall(x)
    # 定量
    if len(rep) > 0:
        p = re.compile(r'(\+)')
        rep = p.findall(x)
        if len(rep) > 1:
            _p = len(rep)
        else:
            _p = 1
#     p = re.compile(r'(阳性|\+|^正常$|Normal)')
    p = re.compile(r'(阴性|[^\d]\-[^\d]|^异常$)')
    rep = p.findall(x)
    # 定量
    if len(rep) > 0:
        p = re.compile(r'(\-)')
        rep = p.findall(x)
        if len(rep) > 1:
            _n = len(rep)
        else:
            _n = 1
#     p = re.compile(r'(阳性|\+|^正常$|Normal)')
    return _p, _n

# def _check_pn(x):
#     p = re.compile(r'(阳性|[^\d]\+[^\d]|^正常$|Normal|^未见$|^未检出$)')
#     rep = p.findall(x)
# #     p = re.compile(r'(阴性|[^0-9]-[^0-9]|^异常$)')
# #     rep = p.findall(x)
#     return rep

def check_p(x):
    # 定型
    if x==None:
        return None
    _p = 0
    p = re.compile(r'(阳性|\+|^正常$|Normal|^未见$|^未检出$)')
    rep = p.findall(x)
    # 定量
    if len(rep) > 0:
        p = re.compile(r'(\+)')
        rep = p.findall(x)
        if len(rep) > 1:
            _p = len(rep)
        else:
            _p = 1
    return _p

def check_n(x):
    # 定型
    if x==None:
        return None
    _p = 0
    p = re.compile(r'(阴性|(\-)|^异常$)')
    rep = p.findall(x)
    # 定量
    if len(rep) > 0:
        p = re.compile(r'[\d]+(\-)[\d]+')
        rep = p.findall(x)
        if len(rep) == 0:
            p = re.compile(r'(\-)')
            rep = p.findall(x)
            if len(rep) > 1:
                _p = len(rep)
            else:
                _p = 1
    return _p


# In[47]:

ls = [u'3301',u'300036',u'300018',u'300019',u'1363',u'3429',u'300005',u'2233',u'2231',u'2229',u'2228',u'459102',u'2230',u'3203',u'300044',u'0425',u'229021',u'3197',u'2282',u'3194',u'3191',u'3192',u'3189',u'3195',u'360',u'459101',u'3190',u'3196',u'3430',u'3485']
new_ls = reduce(lambda x, y:x+y, [[i+'_p', i+'_n', i+'_d'] for i in ls])


# In[48]:

for c in ls:
    train_X_t[c+'_p'] = train_X_t[c].astype(str).apply(check_p)
    train_X_t[c+'_n'] = train_X_t[c].astype(str).apply(check_n)
    train_X_t[c+'_d'] = train_X_t[c].astype(str).apply(hard_convert_float_fina)


# In[49]:

for c in ls:
    test_X[c+'_p'] = test_X[c].astype(str).apply(check_p)
    test_X[c+'_n'] = test_X[c].astype(str).apply(check_n)
    test_X[c+'_d'] = test_X[c].astype(str).apply(hard_convert_float_fina)


# In[50]:

train_X_t_n = train_X_t[new_ls]
test_X_t_n = test_X[new_ls]


# In[51]:

train_X_t_n.insert(loc=0, column='vid', value=train.iloc[:,0])
test_X_t_n.insert(loc=0, column='vid', value=test.iloc[:,0])


# In[52]:

train_X_t_n.to_csv('../data/train_X_d_all_hard_p_b.csv',index=False, sep=',',encoding='utf-8')

test_X_t_n.to_csv('../data/test_X_d_all_hard_p_b.csv',index=False, sep=',',encoding='utf-8')


# In[53]:

print 'data prepare finish'


# In[ ]:




# In[ ]:



