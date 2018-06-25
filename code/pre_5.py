
# coding: utf-8

# In[1]:


import sys
import numpy as np
import pandas as pd
reload(sys)
sys.setdefaultencoding('utf8')
train_y_origin=pd.read_csv('../data/meinian_round1_train_20180408.csv',  sep=',', low_memory=False,encoding='gbk')
train_y_origin=train_y_origin.rename(index=str, columns={u'收缩压':"shousuoya",u'舒张压':"shuzhangya",u'血清甘油三酯':"ganyousanzhi",u'血清高密度脂蛋白':"gaomidu",u'血清低密度脂蛋白':"dimidu"})


# 清洗训练集中的五个指标
def clean_label(x):
    x = str(x)
    if '+' in x:
        i = x.index('+')
        x = x[0:i]
    if '>' in x:
        i = x.index('>')
        x = x[i + 1:]
    if len(x.split('.')) > 2:
        i = x.rindex('.')
        x = x[0:i] + x[i + 1:]
    if '未做' in x or '未查' in x or '弃查' in x:
        x = np.nan
    if str(x).isdigit() == False and len(str(x)) > 4:
        x = x[0:4]
    return x


# 数据清洗
def data_clean(df):
    for c in ["shousuoya", "shuzhangya", "ganyousanzhi", "gaomidu", "dimidu"]:
        # df[c] = df[c].apply(clean_label)
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c] = df[c].fillna(df[c].mean())

    df.loc[(df.shousuoya < 69), 'shousuoya'] = 69
    df.loc[(df.shuzhangya > 148), 'shuzhangya'] = 148
    df.loc[(df.shuzhangya < 37), 'shuzhangya'] = 37
    df.loc[(df.dimidu < 0.08), 'dimidu'] = 0.08
    return df


train_y_origin=data_clean(train_y_origin)

train_y_origin.to_csv("../data/train_y.csv",index=False,encoding="utf-8")

data=pd.read_csv("../data/tmp.csv",low_memory=False)

df=data.describe(include="all")


count=df.loc["count"].sort_values(ascending=False)


colum_name=["0102","2302","0113","0114","0116","1001","0117","0118","3196","0115","0101","0409","0426","0421",
     "0420","0430","0431","0413","0954","0911","0947","0949","0912","0407","0423","0901","0435","1308", "0929", "0432"]



train_data=data[colum_name]


train_data=train_data.fillna("")
train_data["word"]=reduce(lambda x, y: x+y, [train_data.loc[:,item]  for item in train_data.columns.tolist()])


# In[9]:


colum_name=["0102","2302","0113","0114","0116","1001","0117","0118","3196","0115","0101","0409","0426","0421",
     "0420","0430","0431","0413","0954","0911","0947","0949","0912","0407","0423","0901","0435","1308", "0929", "0432","3399","3400","0119","0434"]


origin_data=data[["vid"]+colum_name]
origin_data=origin_data.fillna("")
nlp_data=origin_data[["vid"]].copy()


def deal_common(item):
    if "正常" in str(item) or "无" in str(item) or "未见" in str(item) or "未发现" in str(item):
        return 1
    elif str(item)=="nan":
        return 1
    else:
        return 0


#肾情况
def deal_shen(item):
    if "结石" in str(item) or "钙化" in str(item) or "结晶" in str(item):
        return 0
    else:
        return 1
origin_data["0102_0409_0434"]=origin_data["0102"]+origin_data["0409"]+origin_data["0434"]
nlp_data["shen"]=origin_data["0102_0409_0434"].apply(deal_shen)
nlp_data["shen"].value_counts()


# In[660]:


import re
  
def deal_yanjing(item):
    if  "眼底" in  str(item) and  "出血" in  str(item):
        # print  item
        return   1
    if  "结膜" in  str(item) and ("炎" in  str(item)  or  "结石"  in  str(item)):
        #print  item
        return  1
    if  "玻璃体" in  str(item) or "晶体混浊" in  str(item):
        #print  item
        return   1
    if  "青光" in  str(item):
        #print  item
        return   1
    if  "白内障" in  str(item):
        #print  item
        return   1

    
    if "裸眼" in str(item) or "矫正" in str(item) or "凸光" in str(item):
        #print  item
        return 2
     
    elif "未见明显异常" in str(item) or "未发现明显异常" in str(item) or "未见异常" in str(item) or "眼科检查正常" in str(item) :
        #print  item
        return 2
    elif str(item)=="":
        return 0
    else:
        #print  item
        return 0
nlp_data["yanjing"]=origin_data["1308"].apply(deal_yanjing)
nlp_data["yanjing"].value_counts()


# In[661]:


#血压情况
def deal_xueya(item):
    if "血压" in str(item):
        return 0
    else:
        return 1
# origin_data["0102_0409_0434"]=origin_data["0102"]+origin_data["0409"]+origin_data["0434"]
# nlp_data["xueya"]=origin_data["0102_0409_0434"].apply(deal_xueya)
origin_data["0102_0409"]=origin_data["0102"]+origin_data["0409"]
nlp_data["xueya"]=origin_data["0102_0409"].apply(deal_xueya)
nlp_data["xueya"].value_counts()


# In[662]:


#消化道情况
def deal_xiaohuadao(item):
    if "胃炎" in str(item) or "肠炎" in str(item) or "阑尾炎" in str(item):
        return 0
    else:
        return 1
origin_data["0409_0434"]=origin_data["0409"]+origin_data["0434"]
# nlp_data["xiaohuadao"]=origin_data["0409_0434"].apply(deal_xueya)
# nlp_data["xiaohuadao"].value_counts()


# In[663]:


#血脂情况
def deal_xuezhi(item):
    if "血脂" in str(item):
        return 0
    else:
        return 1
# origin_data["0102_0409"]=origin_data["0102"]+origin_data["0409"]
# nlp_data["xuezhi"]=origin_data["0102_0409"].apply(deal_xuezhi)
origin_data["0102_0409_0434"]=origin_data["0102"]+origin_data["0409"]+origin_data["0434"]
nlp_data["xuezhi"]=origin_data["0102_0409_0434"].apply(deal_xuezhi)
nlp_data["xuezhi"].value_counts()


# In[664]:


#肝情况
def deal_gan(item):
    if "脂肪" in str(item):
        return 1
    else:
        return 0
# origin_data["0102_0409"]=origin_data["0102"]+origin_data["0409"]
# nlp_data["gan"]=origin_data["0102_0409"].apply(deal_gan)

origin_data["0102_0409_0434"]=origin_data["0102"]+origin_data["0409"]+origin_data["0434"]
nlp_data["gan"]=origin_data["0102_0409_0434"].apply(deal_gan)
nlp_data["gan"].value_counts()


# In[665]:


#糖尿病情况
def deal_tangniaobing(item):
    if "糖尿" in str(item):
        #print   item
        return 0
    else:
        return 1
# origin_data["0102_0409"]=origin_data["0102"]+origin_data["0409"]
# nlp_data["tangniaobing"]=origin_data["0102_0409"].apply(deal_tangniaobing)

origin_data["0102_0409_0434"]=origin_data["0102"]+origin_data["0409"]+origin_data["0434"]
nlp_data["tangniaobing"]=origin_data["0102_0409_0434"].apply(deal_tangniaobing)
nlp_data["tangniaobing"].value_counts()


# In[666]:


#动脉硬化情况
def deal_dmyh(item):
    if "双侧颈总动脉" in str(item) and "膜增厚" in str(item):
        #print   item
        return 0
    else:
        return 1
origin_data["0102_0409"]=origin_data["0102"]+origin_data["0409"]
# nlp_data["dmyh"]=origin_data["0102_0409"].apply(deal_dmyh)
# nlp_data["dmyh"].value_counts()


# In[667]:


#检查冠心病情况
def deal_dongmaiyinghu(item):
    if "冠"  in str(item):
        return  2
    else:
        return 1
origin_data["0102_0409"]=origin_data["0102"]+origin_data["0409"]
# nlp_data["dongmaiyinghua"]=origin_data["0102_0409"].apply(deal_dongmaiyinghu)
# nlp_data["dongmaiyinghua"].value_counts()


# In[668]:


#检查血糖情况
def deal_xuetang(item):
    if "血糖"  in str(item):
        return  2
    else:
        return 1
# origin_data["0102_0409"]=origin_data["0102"]+origin_data["0409"]
# nlp_data["xuetang"]=origin_data["0102_0409"].apply(deal_xuetang)


origin_data["0102_0409_0434"]=origin_data["0102"]+origin_data["0409"]+origin_data["0434"]
nlp_data["xuetang"]=origin_data["0102_0409_0434"].apply(deal_xuetang)

nlp_data["xuetang"].value_counts()


# In[669]:


#肥胖情况
def deal_feipang(item):
    if "肥胖" in str(item):
        return 0
    else:
        #print   item
        return 1
origin_data["0102_0409"]=origin_data["0102"]+origin_data["0409"]
nlp_data["feipang"]=origin_data["0102_0409"].apply(deal_feipang)
nlp_data["feipang"].value_counts()


# In[670]:


#甲状腺情况
def deal_jiazhuangxian(item):
    if "结节" in str(item) or "肿大" in str(item) or "略大" in str(item) or "亢进" in str(item) or "甲状腺良性" in str(item) or "甲状腺功能" in str(item):
        return 0
    else:
        return 1
origin_data["0102_0954_0912"]=origin_data["0102"]+origin_data["0954"]+origin_data["0912"]
nlp_data["jiazhuangxian"]=origin_data["0102_0954_0912"].apply(deal_jiazhuangxian)

# origin_data["0102_0954_0912_0409"]=origin_data["0102"]+origin_data["0954"]+origin_data["0912"]+origin_data["0409"]
# nlp_data["jiazhuangxian"]=origin_data["0102_0954_0912_0409"].apply(deal_jiazhuangxian)

nlp_data["jiazhuangxian"].value_counts()


# In[671]:


#心率情况
def deal_xinlv(item):
    if "不齐" in str(item):
        return 4
    elif "过缓" in str(item):
        return 3
    elif "早搏" in str(item):
        return 2
    else:
        return 1
origin_data["1001_0409"]=origin_data["1001"]+origin_data["0409"]
nlp_data["xinlv"]=origin_data["1001_0409"].apply(deal_xinlv)
nlp_data["xinlv"].value_counts()


# In[672]:


#淋巴情况
def deal_linba(item):
    if "无肿大" in str(item) or "不肿大" in str(item):
        return 3
    elif "肿大" in str(item):
        return 2
    else:
        return 1
origin_data["0954_0911"]=origin_data["0954"]+origin_data["0911"]
nlp_data["linba"]=origin_data["0954_0911"].apply(deal_linba)
nlp_data["linba"].value_counts()


# In[673]:


#颈椎情况
def deal_jingzhui(item):
    if "颈椎" in str(item) or "颈项" in str(item) or "腰椎" in str(item):
        return 0
    else:
        return 1
nlp_data["jingzhui"]=origin_data["0947"].apply(deal_jingzhui)
nlp_data["jingzhui"].value_counts()


# In[674]:


#行动情况
def deal_xingdong(item):
    if "活动自如" in str(item) or "未见异常" in str(item) or "活动正常" in str(item):
        #print  item
        return 1
    elif str(item)=="":
        return 1
    else:
        return 0
nlp_data["xingdong"]=origin_data["0949"].apply(deal_xingdong)
nlp_data["xingdong"].value_counts()


# In[675]:


#呼吸情况
def deal_huxi(item):
    if "未见异常" in str(item) or "正常" in str(item):
        return 1
    elif str(item)=="":
        return 1
    else:
        return 0
nlp_data["huxi"]=origin_data["0423"].apply(deal_huxi)
nlp_data["huxi"].value_counts()


# In[676]:


#性别情况
def deal_sex(item):
    if "子宫" in str(item) or "乳" in str(item):
        return 1
    else:
        return 0
nlp_data["sex"]=train_data["word"].apply(deal_sex)
nlp_data["sex"].value_counts()



# In[677]:


#膀胱情况
def deal_pangguang(item):
    if "充盈欠佳" in str(item):
        return 1
    if "充盈良好" in str(item):
        #print  item
        return 0
    if str(item)=="":
        return 2
    if "未见明显" in str(item):
        return 0
    else :
        return 0
nlp_data["deal_pangguang"]=origin_data["0119"].apply(deal_pangguang)
nlp_data["deal_pangguang"].value_counts()
#print  data[["0119"]].head(10)


# In[678]:


def deal_3197(item):
    if "+" in str(item):
        return 0
    else:
        return 1
nlp_data["3197"]=data["3197"].apply(deal_3197)
nlp_data["3197"].value_counts()


# In[679]:


def deal_3400(item):
    if "透明" in str(item):
        return 1
    elif str(item)=="nan":
        return 1
    else:
        return 0
nlp_data["3400"]=data["3400"].apply(deal_3400)
nlp_data["3400"].value_counts()


# In[680]:


def deal_3191(item):
    if "+" in str(item):
        return 0
    else:
        return 1
nlp_data["3191"]=data["3191"].apply(deal_3191)
nlp_data["3191"].value_counts()


# In[681]:


def deal_3194(item):
    if "+" in str(item):
        return 0
    else:
        return 1
nlp_data["3194"]=data["3194"].apply(deal_3194)
nlp_data["3194"].value_counts()


# In[682]:


def deal_3189(item):
    if "+" in str(item):
        return 0
    else:
        return 1
nlp_data["3189"]=data["3189"].apply(deal_3189)
nlp_data["3189"].value_counts()


# In[683]:


def deal_3192(item):
    if "+" in str(item):
        return 0
    else:
        return 1
nlp_data["3192"]=data["3192"].apply(deal_3192)
nlp_data["3192"].value_counts()


# In[684]:


def deal_3190(item):
    if "+" in str(item):
        return 0
    else:
        return 1
nlp_data["3190"]=data["3190"].apply(deal_3190)
nlp_data["3190"].value_counts()


# In[685]:


def deal_0420(item):
    if "弱" in str(item) or "遥远" in str(item):
        return 0
    else:
        return 1
nlp_data["0420"]=data["0420"].apply(deal_0420)
nlp_data["0420"].value_counts()


# In[686]:


def deal_3195(item):
    if "+" in str(item):
        return 0
    else:
        return 1
nlp_data["3195"]=data["3195"].apply(deal_3195)
nlp_data["3195"].value_counts()


# In[687]:


def deal_0440(item):
    if "痛" in str(item):
        return 0
    else:
        return 1
nlp_data["0440"]=data["0440"].apply(deal_0440)
nlp_data["0440"].value_counts()


# In[688]:


def deal_3196(item):
    if "+" in str(item):
        return 0
    else:
        return 1
nlp_data["3196"]=data["3196"].apply(deal_3196)
nlp_data["3196"].value_counts()


# In[689]:


def deal_0425(item):
    if "正常" in str(item) or "未见" in str(item) or "无" in str(item):
        return 16
    try:
        num=int(item)
    except:
        num=16
    return num
# nlp_data["0425"]=data["0425"].apply(deal_0425)
# nlp_data["0425"].value_counts()


# In[690]:


def deal_0216(item):
    if "正常" in str(item) or "未见" in str(item):
        return 1
    elif str(item)=="nan":
        return 1
    else:
        return 0
nlp_data["0216"]=data["0216"].apply(deal_0216)
nlp_data["0216"].value_counts()


# In[691]:


def deal_100010(item):
    if "+" in str(item):
        return 0
    else:
        return 1
nlp_data["100010"]=data["100010"].apply(deal_100010)
nlp_data["100010"].value_counts()


# In[692]:


def deal_30007(item):
    if "Ⅳ" in str(item) or "IV" in str(item) or "Ⅱv" in str(item):
        return 4
    elif "Ⅲ" in str(item) or "III" in str(item) or "iii" in str(item) or "中度" in str(item):
        return 3
    elif "Ⅱ" in str(item) or "II" in str(item) or "ii" in str(item):
        return 2
    elif  str(item)=="nan":
        return 2
    else:
        return 1
nlp_data["30007"]=data["30007"].apply(deal_30007)
nlp_data["30007"].value_counts()


# In[693]:


def deal_3430(item):
    if "+" in str(item):
        return 0
    else:
        return 1
nlp_data["3430"]=data["3430"].apply(deal_3430)
nlp_data["3430"].value_counts()


# In[694]:


def deal_2302(item):
    if "亚健康" in str(item) or "疾病" in str(item) or "肥健康" in str(item) :
        return 0
    else:
        return 1
nlp_data["2302"]=data["2302"].apply(deal_2302)
nlp_data["2302"].value_counts()


# In[695]:


def deal_0707(item):
    if "正常" in str(item) or "未见" in str(item):
        return 1
    elif str(item)=="nan":
        return 1
    else:
        return 0
nlp_data["0707"]=data["0707"].apply(deal_0707)
nlp_data["0707"].value_counts()


# In[696]:


def deal_0901(item):
    if "正常" in str(item) or "未见" in str(item):
        return 1
    elif str(item)=="nan":
        return 1
    else:
        return 0
nlp_data["0901"]=data["0901"].apply(deal_0901)
nlp_data["0901"].value_counts()


# In[697]:


def deal_0436(item):
    if "无" in str(item) or "未见" in str(item) or "不详" in str(item):
        return 1
    elif str(item)=="nan":
        return 1
    else:
        return 0
nlp_data["0436"]=data["0436"].apply(deal_0436)
nlp_data["0436"].value_counts()


# In[698]:


def deal_0439(item):
    if "血压" in str(item) or  "糖" in str(item)  or  "血脂" in str(item)   or  "癌" in str(item) :
        return 0
    else:
        return 1
nlp_data["0439"]=data["0439"].apply(deal_0439)
nlp_data["0439"].value_counts()


# In[699]:


def deal_0202(item):
    if "正常" in str(item) or "未见" in str(item):
        return 1
    elif str(item)=="nan":
        return 1
    else:
        return 0
nlp_data["0202"]=data["0202"].apply(deal_0202)
nlp_data["0202"].value_counts()


# In[700]:


def deal_0217(item):
    if "正常" in str(item) or "未见" in str(item):
        return 1
    elif str(item)=="nan":
        return 1
    else:
        return 0
nlp_data["0217"]=data["0217"].apply(deal_0217)
nlp_data["0217"].value_counts()


# In[701]:


def deal_0706(item):
    if "正常" in str(item) or "未见" in str(item):
        return 1
    elif str(item)=="nan":
        return 1
    else:
        return 0
nlp_data["0706"]=data["0706"].apply(deal_0706)
nlp_data["0706"].value_counts()


# In[702]:


def deal_0215(item):
    if "正常" in str(item) or "未见" in str(item):
        return 1
    elif str(item)=="nan":
        return 1
    else:
        return 0
nlp_data["0215"]=data["0215"].apply(deal_0215)
nlp_data["0215"].value_counts()


# In[703]:


def deal_4001(item):
    if "正常" in str(item) or "良好" in str(item) or "未见" in str(item):
        return 1
    elif str(item)=="nan":
        return 1
    else:
        return 0
nlp_data["4001"]=data["4001"].apply(deal_4001)
nlp_data["4001"].value_counts()


# In[704]:


def deal_0225(item):
    if "正常" in str(item) or "无" in str(item) or "未" in str(item):
        return 1
    elif str(item)=="nan":
        return 1
    else:
        return 0
nlp_data["0225"]=data["0225"].apply(deal_0225)
nlp_data["0225"].value_counts()


# In[705]:


nlp_data["0203"]=data["0203"].apply(deal_common)
nlp_data["0203"].value_counts()


# In[706]:


nlp_data["0209"]=data["0209"].apply(deal_common)
nlp_data["0209"].value_counts()


# In[707]:


def deal_2229(item):
    if "+" in str(item) or "阳" in str(item) or "重" in str(item):
        return 0
    elif str(item)=="nan":
        return 1
    else:
        return 1
nlp_data["2229"]=data["2229"].apply(deal_2229)
nlp_data["2229"].value_counts()


# In[708]:


nlp_data["0705"]=data["0705"].apply(deal_common)
nlp_data["0705"].value_counts()


# In[709]:


nlp_data["1314"]=data["1314"].apply(deal_common)
nlp_data["1314"].value_counts()


# In[710]:


def deal_2231(item):
    if "+" in str(item) or "阳" in str(item) or "重" in str(item):
        return 0
    elif str(item)=="nan":
        return 1
    else:
        return 1
nlp_data["2231"]=data["2231"].apply(deal_2231)
nlp_data["2231"].value_counts()


# In[711]:


nlp_data["1302"]=data["1302"].apply(deal_common)
nlp_data["1302"].value_counts()


# In[712]:


def deal_3601(item):
    if "正常" in str(item):
        return 1
    elif str(item)=="nan":
        return 1
    else:
        return 0
nlp_data["3601"]=data["3601"].apply(deal_3601)
nlp_data["3601"].value_counts()


# In[713]:


nlp_data["1316"]=data["1316"].apply(deal_common)
nlp_data["1316"].value_counts()


# In[714]:


nlp_data["0703"]=data["0703"].apply(deal_common)
nlp_data["0703"].value_counts()


# In[715]:


nlp_data["1402"]=data["1402"].apply(deal_common)
nlp_data["1402"].value_counts()


# In[716]:


def deal_300036(item):
    if "+" in str(item):
        return 0
    elif str(item)=="nan":
        return 1
    else:
        try:
            num=float(str(item))
            if num>10:
                return 0
            else:
                return 1
        except:
            return 1
nlp_data["300036"]=data["300036"].apply(deal_300036)
nlp_data["300036"].value_counts()


# In[717]:


def deal_0509(item):
    if "欠光滑" in str(item) or "肥大" in str(item) or "囊肿" in str(item) or "萎缩小" in str(item):
        return 0
    elif "光滑" in str(item) or "正常" in str(item) or "未见" in str(item) or "未发现" in str(item):
        return 1
    elif str(item)=="nan":
        return 1
    else:
        return 0
nlp_data["0509"]=data["0509"].apply(deal_0509)
nlp_data["0509"].value_counts()


# In[718]:


def deal_3301(item):
    if "阴" in str(item):
        return 1
    elif str(item)=="nan":
        return 1
    else:
        return 0
nlp_data["3301"]=data["3301"].apply(deal_3301)
nlp_data["3301"].value_counts()


# In[719]:


nlp_data["1102"]=data["1102"].apply(deal_common)
nlp_data["1102"].value_counts()


# In[720]:


def deal_0222_yanyan(item):
    if "咽炎" in str(item) or "充血" in str(item)  or "增生" in str(item) or "肿大" in str(item):
        return 0
    elif str(item)=="nan":
        return 1
    else:
        return 1
def deal_0222_biyan(item):
    if "鼻炎" in str(item):
        return 0
    elif str(item)=="nan":
        return 1
    else:
        return 1
nlp_data["0222_yanyan"]=data["0222"].apply(deal_0222_yanyan)
nlp_data["0222_yanyan"].value_counts()
nlp_data["0222_biyan"]=data["0222"].apply(deal_0222_biyan)
nlp_data["0222_biyan"].value_counts()


# In[721]:


list_columns_name=nlp_data.columns.tolist()
lable_name=[u'3301',u'300036',u'300018',u'300019',u'1363',u'3429',u'300005',u'2233',u'2231',u'2229',u'2228',u'459102',u'2230',u'3203',u'300044',u'0425',u'229021',u'3197',u'2282',u'3194',u'3191',u'3192',u'3189',u'3195',u'360',u'459101',u'3190',u'3196',u'3430',u'3485']
# list_columns_name=list(set(list_columns_name)-set(lable_name))
list_columns_name.remove("vid")


# In[722]:


nlp_new_data=nlp_data[["vid"]].copy()
# nlp_new_data=pd.concat([nlp_data[["vid"]],pd.get_dummies(nlp_data[list_columns_name])], axis=1)
for column_name in list_columns_name:
    nlp_new_data = pd.concat([nlp_new_data, pd.get_dummies(nlp_data[column_name], prefix= column_name)], axis=1)


# In[723]:


nlp_new_data.columns = [ name.decode("utf-8") for name in  nlp_new_data.columns.tolist()]
nlp_new_data


# In[724]:


import gc 
gc.collect()


# In[725]:


# train_num_data=pd.read_csv('data/train_X_d.csv',  sep=',', low_memory=False)
train_num_data=pd.read_csv('../data/train_X_d_all_hard_b.csv',  sep=',', low_memory=False)
train_num_data=pd.merge(train_num_data,pd.read_csv('../data/train_X_d_all_hard_p_b.csv',  sep=',', low_memory=False),on=["vid"])
train_y=pd.read_csv('../data/train_y.csv',  sep=',', low_memory=False)
train_num_data=pd.merge(train_num_data,train_y[["vid"]],on=["vid"])
train_data_save=pd.merge(train_num_data,nlp_new_data,on=["vid"])
train_data_save.to_csv("../data/train_data_nlp.csv",index=False,encoding="utf-8")



# In[726]:


test_data_vid=pd.read_csv("../data/meinian_round1_test_b_20180505.csv",low_memory=False,encoding="gbk")
# test_num_data=pd.read_csv("data/test_X_d.csv",low_memory=False)
test_num_data=pd.read_csv("../data/test_X_d_all_hard_b.csv",low_memory=False)
test_num_data=pd.merge(test_num_data,pd.read_csv('../data/test_X_d_all_hard_p_b.csv',  sep=',', low_memory=False),on=["vid"])
test_num_data=pd.merge(test_num_data,test_data_vid[["vid"]],on=["vid"])
test_data_save=pd.merge(test_num_data,nlp_new_data,on=["vid"])
test_data_save.to_csv("../data/test_data_nlp.csv",index=False,encoding="utf-8")



