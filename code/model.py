#coding:utf-8
import math
import numpy as np
import pandas as pd
import lightgbm as lgb
from math import log1p, pow
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def calc_logloss_df(true_df,pred_df):
    flag = (pred_df < 0).any(axis=1)
    print('miss_pred：%f' % (sum(flag)))
    true_df = true_df[~flag]
    pred_df = pred_df[~flag]
    loss_sum = 0
    rows = pred_df.shape[0]
    for c in pred_df.columns:
        # 预测结果必须要>0,否则log函数会报错，导致最终提交结果没有分数
        _true_df = true_df[c].apply(lambda x: log1p(x))
        _pred_df = pred_df[c].apply(lambda x: log1p(x))
        true_ser = (_pred_df - _true_df).apply(lambda x: pow(x, 2))
        loss_item = (true_ser.sum()) / rows
        loss_sum += loss_item
        print c,
        print('的loss：%f' % (loss_item))
    print '总的loss：', loss_sum / len(pred_df.columns)

train = pd.read_csv('../data/train_data_nlp.csv')

test = pd.read_csv('../data/test_data_nlp.csv')

train_target=pd.read_csv('../data/train_y.csv')
train_target=pd.merge(train[["vid"]],train_target,on=["vid"])
train=train.drop(columns=["vid"])
test_id=test.vid
test=test.drop(columns=["vid"])
train_target=train_target.drop(columns=["vid"])

X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(train, train_target, test_size=0.2, random_state=42)

Y_P = pd.DataFrame(data=y_test_p)
Y_P_M = Y_P.copy()
label_c=["shousuoya", "shuzhangya", "ganyousanzhi", "gaomidu", "dimidu"]

column_list=[[u'1850', u'2405', u'1815', u'1115', u'0424', u'10004', u'1345', u'1814', u'2404', u'1117', u'2333', u'192', u'191', u'190', u'2420', u'143', u'2174', u'10003', u'183', u'xueya_0', u'1322', u'2403', u'10002', u'3193', u'1106', u'31', u'1845', u'314', u'sex_man', u'1326', u'300092', u'300017', u'193', u'319', u'39', u'669004', u'979001', u'269008', u'1127', u'269007', u'1321', u'37', u'155', u'2372', u'669001', u'100012', u'2406', u'100007', u'315', u'316', u'shen_0', u'669002', u'313', u'269004', u'1474', u'1840', u'269018', u'gan_0', u'300012', u'269011', u'979017', u'1325', u'669005', u'979011', u'809009', u'100006', u'320', u'100005', u'269006', u'669006', u'10009', u'269012', u'1107', u'809004', u'300021', u'139', u'269020', u'269025', u'2177', u'300011', u'979005', u'jiazhuangxian_0', u'100014', u'979003', u'269010', u'979008', u'300001', u'2376', u'809021', u'33', u'979009', u'100013', u'269021', u'269017', u'300013', u'2386', u'269024', u'269009', u'979006', u'ruxian_0', u'1112', u'269005', u'979023', u'312', u'669009', u'269003', u'32', u'317', u'269014', u'979004', u'xinlv_2', u'xinlv_4', u'xingdong_0', u'979012', u'809025', u'809017', u'269013', u'979002', u'38', u'34', u'269022', u'269016', u'979021', u'809026', u'tangniaobing_0', u'809001', u'269023', u'809023', u'269015', u'269019', u'979016', u'xinlv_1', u'979013', u'yanjing_1', u'979015', u'979014', u'979019', u'979007', u'979020', u'809008', u'yanjing_0', u'979022', u'300008', u'979018', u'xueya_1', u'gan_1', u'shen_1', u'jiazhuangxian_1', u'xingdong_1', u'sex_woman', u'ruxian_1', u'xinlv_3', u'linba_2', u'linba_3', u'yanjing_2', u'jingzhui_0', u'huxi_0', u'xuezhi_0', u'xuezhi_1', u'tangniaobing_1', u'feipang_0', u'feipang_1', u'linba_1', u'jingzhui_1', u'huxi_1'], [u'1850', u'2405', u'0424', u'1117', u'10004', u'2403', u'1815', u'2420', u'1115', u'2333', u'190', u'313', u'10002', u'xueya_0', u'1814', u'191', u'1345', u'31', u'100005', u'183', u'2372', u'2404', u'314', u'192', u'300092', u'269012', u'10003', u'1845', u'sex_man', u'193', u'39', u'269011', u'979012', u'2174', u'1107', u'1106', u'669005', u'316', u'xinlv_1', u'37', u'669001', u'10009', u'669009', u'100007', u'2406', u'979006', u'155', u'3193', u'1127', u'979011', u'143', u'tangniaobing_0', u'269018', u'1326', u'269013', u'2177', u'1840', u'669004', u'300017', u'317', u'979005', u'100006', u'32', u'269007', u'300021', u'319', u'269025', u'1321', u'320', u'269008', u'1322', u'979001', u'315', u'100013', u'809021', u'979003', u'269009', u'100012', u'gan_0', u'669006', u'1325', u'2386', u'jiazhuangxian_0', u'312', u'1474', u'809001', u'33', u'269003', u'269019', u'269006', u'2376', u'300008', u'669002', u'979007', u'269004', u'38', u'300012', u'300011', u'979016', u'ruxian_0', u'100014', u'979004', u'shen_0', u'979009', u'269005', u'269024', u'979008', u'809009', u'809004', u'269022', u'269014', u'979013', u'300001', u'809025', u'979002', u'269020', u'269021', u'269017', u'809017', u'269023', u'269016', u'979021', u'809008', u'809023', u'300013', u'139', u'34', u'269015', u'979015', u'809026', u'xinlv_2', u'269010', u'979023', u'979020', u'979019', u'1112', u'979018', u'huxi_0', u'yanjing_1', u'xingdong_0', u'979022', u'979014', u'yanjing_0', u'xueya_1', u'tangniaobing_1', u'xinlv_3', u'sex_woman', u'979017', u'yanjing_2', u'jingzhui_0', u'shen_1', u'ruxian_1', u'xuezhi_0', u'xuezhi_1', u'gan_1', u'feipang_0', u'feipang_1', u'jiazhuangxian_1', u'xinlv_4', u'linba_1', u'linba_2', u'linba_3', u'jingzhui_1', u'xingdong_1', u'huxi_1'], [u'1850', u'1117', u'193', u'191', u'183', u'192', u'1814', u'10004', u'2405', u'1815', u'10003', u'2403', u'1115', u'269008', u'10002', u'269009', u'190', u'0424', u'1345', u'2372', u'313', u'979001', u'314', u'2174', u'1107', u'2404', u'319', u'2406', u'1840', u'317', u'1845', u'2333', u'2420', u'143', u'39', u'300021', u'979011', u'100006', u'269003', u'269012', u'979023', u'1106', u'33', u'979015', u'gan_0', u'312', u'100005', u'xuezhi_0', u'300017', u'979007', u'320', u'979008', u'1322', u'139', u'269016', u'979021', u'100007', u'2177', u'31', u'1127', u'979012', u'669009', u'1112', u'xinlv_1', u'32', u'38', u'269018', u'269025', u'979017', u'300092', u'669006', u'809023', u'300001', u'10009', u'979002', u'809004', u'sex_man', u'300011', u'1474', u'979005', u'669001', u'1321', u'269020', u'315', u'269011', u'669005', u'3193', u'34', u'316', u'269014', u'100013', u'shen_0', u'2376', u'100014', u'300008', u'269004', u'979009', u'979022', u'979004', u'100012', u'300012', u'269007', u'1325', u'269010', u'979006', u'669004', u'155', u'979003', u'269017', u'269006', u'979014', u'979018', u'979019', u'809009', u'yanjing_2', u'1326', u'269013', u'2386', u'269005', u'37', u'269015', u'979016', u'xuezhi_1', u'809021', u'300013', u'269024', u'269021', u'269023', u'979020', u'809008', u'gan_1', u'jiazhuangxian_0', u'809001', u'269022', u'269019', u'809017', u'809026', u'xueya_0', u'linba_1', u'669002', u'yanjing_0', u'xueya_1', u'xinlv_3', u'linba_2', u'xingdong_0', u'809025', u'979013', u'shen_1', u'ruxian_0', u'ruxian_1', u'yanjing_1', u'tangniaobing_0', u'tangniaobing_1', u'feipang_0', u'feipang_1', u'jiazhuangxian_1', u'xinlv_2', u'xinlv_4', u'linba_3', u'jingzhui_0', u'jingzhui_1', u'xingdong_1', u'huxi_0', u'huxi_1', u'sex_woman'], [u'1106', u'1815', u'1117', u'10004', u'1850', u'0424', u'1814', u'2372', u'100005', u'191', u'2174', u'2405', u'190', u'193', u'2420', u'2403', u'192', u'39', u'314', u'2386', u'100007', u'183', u'2406', u'1115', u'300021', u'10002', u'269007', u'315', u'2333', u'269008', u'linba_3', u'1345', u'1845', u'2404', u'979009', u'300012', u'100006', u'100013', u'300092', u'979006', u'31', u'269011', u'979023', u'320', u'269003', u'1107', u'312', u'669002', u'269025', u'10003', u'317', u'300017', u'300013', u'269009', u'32', u'1840', u'34', u'979005', u'269013', u'316', u'gan_0', u'1127', u'sex_man', u'809001', u'269005', u'669001', u'269010', u'979017', u'300008', u'155', u'3193', u'269020', u'100014', u'143', u'38', u'2376', u'33', u'269014', u'669009', u'37', u'269022', u'269016', u'269004', u'979007', u'269012', u'319', u'139', u'100012', u'979012', u'10009', u'1474', u'269019', u'979018', u'809009', u'1322', u'979001', u'313', u'809025', u'269015', u'300011', u'huxi_0', u'269006', u'979021', u'2177', u'1321', u'669004', u'jingzhui_0', u'979002', u'269021', u'669006', u'269023', u'1325', u'979004', u'809023', u'1112', u'669005', u'300001', u'yanjing_2', u'jiazhuangxian_0', u'1326', u'809021', u'979015', u'979019', u'linba_1', u'979003', u'979008', u'809004', u'809026', u'269024', u'979020', u'979014', u'809008', u'809017', u'yanjing_1', u'yanjing_0', u'979016', u'979011', u'269018', u'979022', u'979013', u'shen_0', u'gan_1', u'269017', u'xinlv_3', u'jiazhuangxian_1', u'jingzhui_1', u'xingdong_0', u'huxi_1', u'sex_woman', u'shen_1', u'ruxian_0', u'xuezhi_0', u'tangniaobing_0', u'xinlv_1', u'ruxian_1', u'xueya_0', u'xueya_1', u'xuezhi_1', u'tangniaobing_1', u'feipang_0', u'feipang_1', u'xinlv_2', u'xinlv_4', u'linba_2', u'xingdong_1'], [u'193', u'1107', u'1850', u'1117', u'10004', u'10002', u'192', u'190', u'2406', u'100007', u'191', u'314', u'2333', u'1815', u'2405', u'2372', u'2174', u'269011', u'2420', u'319', u'2404', u'0424', u'1115', u'100006', u'100005', u'320', u'269003', u'183', u'1814', u'33', u'2403', u'269007', u'10003', u'39', u'38', u'37', u'1845', u'269006', u'313', u'1345', u'1112', u'300021', u'317', u'316', u'669004', u'979001', u'269008', u'300017', u'300092', u'31', u'315', u'34', u'669009', u'269004', u'809021', u'139', u'32', u'100013', u'300001', u'269013', u'1127', u'979017', u'1322', u'143', u'1840', u'269019', u'3193', u'809001', u'269025', u'979005', u'10009', u'100012', u'100014', u'979004', u'2177', u'669006', u'155', u'1321', u'1474', u'312', u'979006', u'979012', u'979007', u'300012', u'linba_3', u'300011', u'269009', u'269016', u'1106', u'2386', u'979009', u'669002', u'979003', u'669005', u'979022', u'809017', u'xuezhi_0', u'300008', u'669001', u'809008', u'269010', u'269005', u'979020', u'300013', u'979002', u'269012', u'979019', u'809023', u'979008', u'269018', u'269021', u'809009', u'269014', u'979023', u'gan_0', u'269023', u'979021', u'1325', u'269020', u'269024', u'269017', u'979014', u'269015', u'809004', u'269022', u'979015', u'yanjing_0', u'jiazhuangxian_0', u'1326', u'tangniaobing_0', u'2376', u'979011', u'979013', u'yanjing_2', u'979016', u'979018', u'shen_0', u'linba_1', u'jingzhui_0', u'809025', u'xinlv_1', u'809026', u'xuezhi_1', u'sex_man', u'xueya_0', u'xinlv_2', u'xinlv_4', u'yanjing_1', u'shen_1', u'jiazhuangxian_1', u'jingzhui_1', u'xingdong_0', u'ruxian_0', u'ruxian_1', u'xueya_1', u'gan_1', u'tangniaobing_1', u'feipang_0', u'feipang_1', u'xinlv_3', u'linba_2', u'xingdong_1', u'huxi_0', u'huxi_1', u'sex_woman']]
column_num=100

column_list=[]

'''
for index,c_n in enumerate(label_c) :

    # lgb_train = lgb.Dataset(X_train_p[column_list[index][:column_num]], y_train_p[c_n])
    # lgb_eval = lgb.Dataset(X_test_p[column_list[index][:column_num]], y_test_p[c_n], reference=lgb_train)

    lgb_train = lgb.Dataset(X_train_p, y_train_p[c_n])
    lgb_eval = lgb.Dataset(X_test_p, y_test_p[c_n], reference=lgb_train)
    # params = {
    #     'task': 'train',
    #     'boosting_type': 'gbdt',
    #     'objective': 'regression',
    #     #     'metric': {'l2', 'auc'},
    #     'num_leaves': 31,
    #     'learning_rate': 0.05,
    #     'feature_fraction': 0.9,
    #     'bagging_fraction': 0.8,
    #     'bagging_freq': 5,
    #     'seed': 0,
    #     'verbose': 1
    # }
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'regression',
        #     'metric': {'l2', 'auc'},
        'num_leaves': 110,
        'learning_rate': 0.005,
        'n_estimators': 3500,
        'min_data_in_leaf': 20,
        'max_depth': 12,
        'subsample': 0.8,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'seed': 0,
        'verbose': 1
    }
    print('Start training...')
    # train
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=500,
                    valid_sets=lgb_eval,
                    early_stopping_rounds=10)
    # y_pred = gbm.predict(X_test_p[column_list[index][:column_num]], num_iteration=gbm.best_iteration)
    y_pred = gbm.predict(X_test_p, num_iteration=gbm.best_iteration)
    m_pred = [train_target[c_n].mean()] * len(y_pred)
    testScore = math.sqrt(mean_squared_error(y_test_p[c_n].values, y_pred))
    testScore_b = math.sqrt(mean_squared_error(y_test_p[c_n].values, m_pred))
    # [train_Y[label_c[0]].mean()] * len(train_Y)
    print(testScore), testScore_b
    Y_P_M[c_n] = y_pred
    # print list(gbm.feature_importance())
    # print gbm.feature_name()
    feature_list=sorted(zip(list(gbm.feature_name()),list(gbm.feature_importance())),key=lambda x :x[1])
    print feature_list
    column_list.append([item[0] for item in feature_list])
print column_list




calc_logloss_df(Y_P, Y_P_M)


exit(0)

'''

'''
生成提交数据
'''

train_X=train
train_Y=train_target
test_X=test
Y_P_M = []
for _i in range(len(label_c)):
    c_n = label_c[_i]
    # lgb_train = lgb.Dataset(train_X[column_list[_i][:column_num]], train_Y[c_n])
    lgb_train = lgb.Dataset(train_X, train_Y[c_n])
    # params = {
    #     'task': 'train',
    #     'boosting_type': 'gbdt',
    #     'objective': 'regression',
    #     #     'metric': {'l2', 'auc'},
    #     'num_leaves': 31,
    #     'learning_rate': 0.05,
    #     'feature_fraction': 0.9,
    #     'bagging_fraction': 0.8,
    #     'bagging_freq': 5,
    #     'seed': 0,
    #     'verbose': 1
    # }
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'regression',
        #     'metric': {'l2', 'auc'},
        'num_leaves': 110,
        'learning_rate': 0.005,
        'n_estimators': 3500,
        'min_data_in_leaf': 20,
        'max_depth': 12,
        'subsample': 0.8,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'seed': 0,
        'verbose': 1
    }
    print('Start training...')
    # train
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=300)
    y_pred = gbm.predict(test_X, num_iteration=gbm.best_iteration)
    m_pred = [train_Y[c_n].mean()] * len(y_pred)
    Y_P_M.append(y_pred)

Y_P_M_ = pd.DataFrame(data=np.array(Y_P_M).T, index=test_id, columns=label_c)
import time
t=time.localtime()
t_str=time.strftime("_%Y%d%m_%H%M%S",t)
Y_P_M_.to_csv('../submit/submit'+t_str+'.csv',index=True,header=False, sep=',',encoding='utf-8')

