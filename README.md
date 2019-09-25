# 01classify

## Reuqirments
* Tensorflow-gpu==1.12.0
* Download [BERT-Base中文模型](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip) or [Chinese-BERT-wwm](https://github.com/ymcui/Chinese-BERT-wwm)


## Usage
1. Format (seperate by '\t' rather than ',') dataset like:
    ````
    train.txt:
    1   ChineseTextChineseText
    0   ChineseTextChineseTextChineseText
    0   ChineseText
    1   ChineseTextChineseTextChineseTextChineseText
    
    eval.txt:
    1   ChineseTextChineseText
    0   ChineseTextChineseTextChineseText
    0   ChineseText
    1   ChineseTextChineseTextChineseTextChineseText
    
    test.txt
    0   ChineseTextChineseTextChineseText
    0   ChineseTextChineseTextChineseText
    0   ChineseTextChineseTextChineseText
    0   ChineseTextChineseTextChineseText
    ````
2. Set runtime paraments in `arguments.py`. 
3. training: 'do_predict' set to **False**, 'do_train' and 'do_eval' set to **True** in `arguments.py`, and run `train_eval.py`.
4. predict: 'do_predict' set to **True**, 'do_train' and 'do_eval' set to **False** in `arguments.py`, and run `train_eval.py`.
5. Find the results file in `./output/test_results.tsv`.
