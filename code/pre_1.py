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


# In[3]:

# 读取数据
train=pd.read_csv('../data/meinian_round1_train_20180408.csv',sep=',',encoding='gbk')
# test=pd.read_csv('data/meinian_round1_test_a_20180409.csv',sep=',',encoding='gbk')
test=pd.read_csv('../data/meinian_round1_test_b_20180505.csv',sep=',',encoding='gbk')

data_part1=pd.read_csv('../data/meinian_round1_data_part1_20180408.txt',sep='$',encoding='utf-8')
data_part2=pd.read_csv('../data/meinian_round1_data_part2_20180408.txt',sep='$',encoding='utf-8')


# In[4]:

# data_part1和data_part2进行合并，并剔除掉与train、test不相关vid所在的行
part1_2 = pd.concat([data_part1,data_part2],axis=0)#{0/'index', 1/'columns'}, default 0
part1_2 = pd.DataFrame(part1_2).sort_values('vid').reset_index(drop=True)
vid_set=pd.concat([train['vid'],test['vid']],axis=0)
vid_set=pd.DataFrame(vid_set).sort_values('vid').reset_index(drop=True)
part1_2=part1_2[part1_2['vid'].isin(vid_set['vid'])]


# In[5]:

# 根据常识判断无用的'检查项'table_id，过滤掉无用的table_id
def filter_None(data):
#     data=data[data['field_results']!='']
    data=data[data['field_results']!='未查']
    data=data[data['field_results']!='未检']
    
    data=data[data['field_results']!='弃查']
    data=data[data['field_results']!='详见报告单']
    data=data[data['field_results']!='详见报告']

    return data

part1_2=filter_None(part1_2)
part1_2['field_results'] = part1_2['field_results'].astype(str)


# In[6]:

vid_tabid_group = part1_2.groupby(['vid','table_id']).size().reset_index()


# In[7]:

vid_tabid_group['new_index'] = vid_tabid_group['vid'] + '_' + vid_tabid_group['table_id']
vid_tabid_group_dup = vid_tabid_group[vid_tabid_group[0]>1]['new_index']


# In[8]:

part1_2['new_index'] = part1_2['vid'] + '_' + part1_2['table_id']


# In[9]:

# 重复数据的拼接操作
def merge_table(df):
#     df['field_results'] = df['field_results'].astype(str)
    if df.shape[0] > 1:
        merge_df = "|".join(list(df['field_results']))
    else:
        merge_df = df['field_results'].values[0]
    return merge_df


dup_part = part1_2[part1_2['new_index'].isin(list(vid_tabid_group_dup))]
dup_part = dup_part.sort_values(['vid','table_id'])
unique_part = part1_2[~part1_2['new_index'].isin(list(vid_tabid_group_dup))]

part1_2_dup = dup_part.groupby(['vid','table_id']).apply(merge_table).reset_index()
part1_2_dup.rename(columns={0:'field_results'},inplace=True)
part1_2_res = pd.concat([part1_2_dup,unique_part[['vid','table_id','field_results']]])

table_id_group=part1_2.groupby('table_id').size().sort_values(ascending=False)


# In[10]:

merge_part1_2 = part1_2_res.pivot(index='vid',values='field_results',columns='table_id')


# In[11]:

remain_part_1_2 = merge_part1_2[:]


# In[12]:

train_of_part=remain_part_1_2[remain_part_1_2.index.isin(train['vid'])]
test_of_part=remain_part_1_2[remain_part_1_2.index.isin(test['vid'])]


# In[13]:

n_train=pd.merge(train,train_of_part,right_index=True, left_on='vid')
n_test=pd.merge(test,test_of_part,right_index=True, left_on='vid')


# In[14]:

# 清洗训练集中的五个指标
def clean_label(x):
    x=str(x)
    if '+' in x:#16.04++
        i=x.index('+')
        x=x[0:i]
    if '>' in x:#> 11.00
        i=x.index('>')
        x=x[i+1:]
    if len(x.split('.'))>2:#2.2.8
        i=x.rindex('.')
        x=x[0:i]+x[i+1:]
    if '未做' in x or '未查' in x or '弃查' in x:
        x=np.nan
    if str(x).isdigit()==False and len(str(x))>4:
        x=x[0:4]
    return x

# 数据清洗
def data_clean(df):
    for c in [u'收缩压',u'舒张压',u'血清甘油三酯',u'血清高密度脂蛋白',u'血清低密度脂蛋白']:
        df[c]=df[c].apply(clean_label)
        df[c]=df[c].astype('float64')
    return df

f_n_train=data_clean(n_train)


# In[15]:

# f_n_train.to_csv('data/tmp/n_train.csv',index=False, sep=',',encoding='utf-8')
# n_test.to_csv('data/tmp/n_test.csv',index=False,  sep=',',encoding='utf-8')

f_n_train.to_csv('../data/n_train_all_b.csv',index=False, sep='$',encoding='utf-8')
n_test.to_csv('../data/n_test_all_b.csv',index=False,  sep='$',encoding='utf-8')


# In[ ]:
