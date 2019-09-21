# -*- coding:utf-8 -*-

'''

mainly classification function realization

'''

import pandas as pd
from collections import Counter
import numpy as np
from sklearn import svm
from sklearn.metrics import f1_score
import jieba
jieba.initialize()  # 手动初始化
from sklearn.externals import joblib

# 构造停用词词典
def MakeStopDic():
    stopDic = {}
    fopen = open('./stopwords.txt', 'r', encoding='utf-8')
    for eachLine in fopen:
        strText = ""
        strText += eachLine  # 解码，使停用词典可用
        # 去掉空格、换行符
        strText = strText.strip()
        strText = strText.replace('\r', '')
        strText = strText.replace('\n', '')
        stopDic[strText] = 1
    fopen.close()
    return stopDic

def GetQuestionValue(str):
    v = Counter(str)['?']
    if v in [1, 3]:
        return v
    elif v >= 3:
        return 3
    else:
        return 0

def GetLengthValue(strs):
    if strs[0] < strs[1]:   # if description is longer than title, then this feature set 0
        return 0
    l = 0       # statics number of chinese characters
    str = strs[0]
    for char in str:
        if '\u4e00' <= char <= '\u9fff':
            l += 1
    if l <= 7:
        return 1
    return 0

def GetRefitValue(strs):
    v = 0
    refit_words = ['改装', '加装']
    for word in refit_words:        # check title whether exist refit related-words
        if Counter(strs[0])[word]:
            v += 1
    for word in refit_words:        # check description whether exist refit related-words
        if Counter(strs[1])[word]:
            v += 1
    if v > 0:
        return 1
    return 0

def TF_IDF(data, dict):
    # ------- generate words list ---------
    segList = []    # store all words
    stopDic = MakeStopDic()  # get stop dictionary
    # concatenate string
    for strText in data:
        seg = []
        # delete residual symbol
        strText = strText.strip()
        strText = strText.replace(' ', '')
        strText = strText.replace('\n', '')
        # separate word with chinese cut tools
        segRes = jieba.cut(strText, cut_all=False)
        # add word to segList
        for word in segRes:
            if word not in stopDic:
                seg.append(word)
        segList.append(seg)
    # ------- generate word vector --------
    _index, cnt = 1, len(dict)
    vecSet = np.zeros((data.shape[0]+1, 100000))  # 向量表集合
    dicFileNum = np.zeros((100000))  # 表示某个词在多少个文档里面出现过
    for reg in segList:
        vis = np.zeros((100000))
        for word in reg:
            if word in dict:
                ind = dict[word]
                vecSet[_index, ind] += 1
                if vis[ind] == 0:
                    dicFileNum[ind] += 1
                    vis[ind] = 1
            else:
                print("!!!!!!")
        _index += 1
    print(f'The number of word is {cnt}')
    # --------- generate TD-IDF ---------
    TF_IDF = np.zeros((data.shape[0] + 1, cnt))
    for i in range(1, data.shape[0]+1):
        for j in range(cnt):
            TF_IDF[i, j] = vecSet[i, j] * np.log(float(vecSet.shape[0]-1)/(dicFileNum[j]+1))
    # sort TF_IDF
    sort_TFIDF = np.zeros((data.shape[0] + 1, 10))
    for i in range(1, data.shape[0] + 1):
        td_idf = TF_IDF[i, :]
        sort_TFIDF[i, :] = np.argsort(-td_idf)[:10]  # 降序排列的索引号，取前十个
    return sort_TFIDF[1:]

def FeatureExtractor(data, dict):
    # extract question feature
    question_vector = [[], []]
    question_vector[0] = [GetQuestionValue(s) for s in data[:, 0]]
    question_vector[1] = [GetQuestionValue(s) for s in data[:, 1]]
    # string total length
    is_too_short = [GetLengthValue(s) for s in data]
    # is 'Refit' words occur in data
    is_exist_refit = [GetRefitValue(s) for s in data]
    # TF-IDF feature
    title_tfidf = TF_IDF(data[:, 0], dict).T
    descr_tfidf = TF_IDF(data[:, 1], dict).T
    print(title_tfidf.shape, descr_tfidf.shape)
    print(np.array(question_vector).shape, np.array(is_too_short).shape, np.array(is_exist_refit).shape)
    print(descr_tfidf)
    features = np.concatenate((question_vector, title_tfidf, descr_tfidf, [is_too_short], [is_exist_refit]))
    return features.T

class Model:
    def __init__(self, train_data, test_data, model=None):
        self.tr_data = train_data
        self.te_data = test_data
        self.dict = {}
        self.dict = self.GetWordDic(self.dict, train_data[:, 2])
        self.dict = self.GetWordDic(self.dict, train_data[:, 3])
        self.dict = self.GetWordDic(self.dict, test_data[:, 1])
        self.dict = self.GetWordDic(self.dict, test_data[:, 2])
        if model:
            self.clf = joblib.load(model)   # load model

    def GetWordDic(self, dict, data):
        # ------- generate words list ---------
        segList = []  # store all words
        stopDic = MakeStopDic()  # get stop dictionary
        # concatenate string
        for strText in data:
            seg = []
            # delete residual symbol
            strText = strText.strip()
            strText = strText.replace(' ', '')
            strText = strText.replace('\n', '')
            # separate word with chinese cut tools
            segRes = jieba.cut(strText, cut_all=False)
            # add word to segList
            for word in segRes:
                if word not in stopDic:
                    seg.append(word)
            segList.append(seg)
        # ------- generate word vector --------
        cnt = len(dict)
        for reg in segList:
            for word in reg:
                if word not in dict:
                    dict[word] = cnt
                    cnt += 1
        print(f'Now the number of word is {cnt}.')
        return dict

    def Training(self):
        features = FeatureExtractor(self.tr_data[:, 2:], self.dict)
        label = self.tr_data[:, 1].astype('int')
        print(features, label)
        clf = svm.SVC(kernel='rbf', decision_function_shape='ovr')
        #print(Counter(label)[0], Counter(label)[1], Counter(label)[0]+Counter(label)[1])
        clf.fit(features, label)  # train
        pre_label = clf.predict(features)  # predict train dataset
        #print(Counter(pre_label)[0], Counter(pre_label)[1])
        print(f"训练集准确率:{float(sum(pre_label == label)) / len(label)}")
        print(f"训练集F1-Score:{f1_score(label, pre_label)}")
        #y_hat = clf.predict(x_test)  # predict test dataset
        joblib.dump(clf, "./svm_for_poker.pkl")  # save model
        self.clf = clf

    def Classify(self):
        # predict with svm
        features = FeatureExtractor(self.te_data[:, 1:], self.dict)
        result = self.te_data[:, :2]    # initial result
        result[:, 1] = self.clf.predict(features)
        # # base on rule
        # cnt = 0
        # for strs in self.te_data[:, 1:]:
        #     if self.BaseRule(strs) == 0:
        #         result[cnt, 1] = 0
        #     cnt += 1
        # save file
        df = pd.DataFrame(result).rename(columns={'0': 'id', '1': 'flag'})
        df.to_csv('results.csv', index=False)

    def BaseRule(self, strs):
        # if title and description have chinese words less than 6, then label 0. otherwise return -1.
        l = 0
        str = ''.join(strs[0]) + ''.join(strs[1])
        for s in str:
            if '\u4e00' <= s <= '\u9fff':
                l += 1
        if l > 6:
            return -1
        return 0

if __name__ == '__main__':
    train_data = pd.read_csv('new_train.csv').fillna('0').values
    test_data = pd.read_csv('new_test.csv').fillna('0').values
    model = Model(train_data, test_data, 'svm_for_poker.pkl')
    # model.Training()
    model.Classify()