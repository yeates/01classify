# -*- coding:utf-8 -*-
'''
input and preprocess the data.
'''
import pandas as pd
import numpy as np
import re
from UselessWords import *
import logging

class DataInputer:
    def __init__(self, tr_data_file, te_data_file):
        self.raw_tr_data = pd.read_csv(tr_data_file).fillna('0').values
        self.raw_te_data = pd.read_csv(te_data_file).fillna('0').values
        # format data in general way
        self.initial_format()

        # firstly handle the train data.
        for i in range(self.raw_tr_data.shape[0]):
            self.clear_useless_words(i, 'train')
        # then handle the test data
        for i in range(self.raw_te_data.shape[0]):
            self.clear_useless_words(i, 'test')

        # ending format
        self.ending_format()

        # save data
        print(self.raw_tr_data)
        pd.DataFrame(self.raw_tr_data).to_csv('./new_train.csv', encoding="utf_8_sig", index=False)
        pd.DataFrame(self.raw_te_data).to_csv('./new_test.csv', encoding="utf_8_sig", index=False)

    def initial_format(self):
        self.raw_tr_data[:, 2] = [s.strip() for s in self.raw_tr_data[:, 2]]
        self.raw_tr_data[:, 3] = [s.strip() for s in self.raw_tr_data[:, 3]]
        self.raw_te_data[:, 1] = [s.strip() for s in self.raw_te_data[:, 1]]
        self.raw_te_data[:, 2] = [s.strip() for s in self.raw_te_data[:, 2]]

    def ending_format(self):
        self.replace_str(u'\u3000', u'')
        self.replace_str('？', '?')
        self.delete_by_regular('(&#)\S.*?(;)')   # illegal symbols
        self.delete_by_regular('[\dA-Za-z]{10,30}')
        r = "[.!/_,$&\-;%^*()<>+""'?@|:~{}#]+|[——！\\\，。=？、：“”‘’￥……（）－《》【】]"
        self.delete_by_regular(r)
        self.replace_str(' ', '')
        self.delete_by_regular('[\dA-Za-z]{10,30}')
        self.delete_by_regular('(\[)\S*?(\])')
        self.delete_by_regular('(此主帖已经被)\S*?(设为精华帖)',)   # No use information

    def delete_by_regular(self, r):
        self.raw_tr_data[:, 2] = [re.sub(r, '', s) for s in self.raw_tr_data[:, 2]]
        self.raw_tr_data[:, 3] = [re.sub(r, '', s) for s in self.raw_tr_data[:, 3]]
        self.raw_te_data[:, 1] = [re.sub(r, '', s) for s in self.raw_te_data[:, 1]]
        self.raw_te_data[:, 2] = [re.sub(r, '', s) for s in self.raw_te_data[:, 2]]

    def replace_str(self, waited, target):
        self.raw_tr_data[:, 2] = [s.replace(waited, target) for s in self.raw_tr_data[:, 2]]
        self.raw_tr_data[:, 3] = [s.replace(waited, target) for s in self.raw_tr_data[:, 3]]
        self.raw_te_data[:, 1] = [s.replace(waited, target) for s in self.raw_te_data[:, 1]]
        self.raw_te_data[:, 2] = [s.replace(waited, target) for s in self.raw_te_data[:, 2]]

    def clear_useless_words(self, row, flag):
        # delete operations
        for str in OPERATIONS:
            self.delete_str(row, str, flag)
        # delete illegal symbol
        for str in ILLEGAL_SYMBOL:
            self.delete_str(row, str, flag)
        # delete by regular expression
        strs0 = []
        for reg in ALL_REGS:
            strs0 = self.find_str_by_reg(reg, row, flag)
        for str in strs0:
            self.delete_str(row, str, flag)

    def delete_str(self, row, str, flag):
        if flag == 'train':
            self.raw_tr_data[row, 2] = self.raw_tr_data[row, 2].replace(str, '')
            self.raw_tr_data[row, 3] = self.raw_tr_data[row, 3].replace(str, '')
        elif flag == 'test':
            self.raw_te_data[row, 1] = self.raw_te_data[row, 1].replace(str, '')
            self.raw_te_data[row, 2] = self.raw_te_data[row, 2].replace(str, '')

    def find_str_by_reg(self, reg, row, flag):
        strs = []
        if flag == 'train':
            strs_1 = np.array(re.findall(reg, self.raw_tr_data[row, 2]))
            strs_2 = np.array(re.findall(reg, self.raw_tr_data[row, 3]))
            if strs_1.shape[0] == 1:
                print(strs_1, reg)
                if len(strs_1.shape) == 1:
                    strs = np.concatenate((strs, strs_1))
                if len(strs_1.shape) > 1:
                    strs = np.concatenate((strs, strs_1[0]))
            if strs_2.shape[0] == 1:
                print(strs_2, reg)
                if len(strs_2.shape) == 1:
                    strs = np.concatenate((strs, strs_2))
                if len(strs_2.shape) > 1:
                    strs = np.concatenate((strs, strs_2[0]))
        elif flag == 'test':
            strs_1 = np.array(re.findall(reg, self.raw_te_data[row, 1]))
            strs_2 = np.array(re.findall(reg, self.raw_te_data[row, 2]))
            if strs_1.shape[0] == 1:
                if len(strs_1.shape) == 1:
                    strs = np.concatenate((strs, strs_1))
                if len(strs_1.shape) > 1:
                    strs = np.concatenate((strs, strs_1[0]))
            if strs_2.shape[0] == 1:
                if len(strs_2.shape) == 1:
                    strs = np.concatenate((strs, strs_2))
                if len(strs_2.shape) > 1:
                    strs = np.concatenate((strs, strs_2[0]))
        return strs


if __name__ == '__main__':
    DI = DataInputer('./data/train.csv', './data/test.csv')
