
import pandas as pd

# train = pd.read_csv('./old_version/data/new_train.csv').fillna('-')
# test = pd.read_csv('./old_version/data/new_test.csv').fillna('-')
#
# train.loc[:, 'full_text'] = [str(a) + str(b) for a, b in zip(train.title, train.content)]
# test.loc[:, 'full_text'] = [str(a) + str(b) for a, b in zip(test.title, test.content)]
#
# test.loc[:, 'flag'] = 1
#
# train = train[['flag', 'full_text']]
# test = test[['flag', 'full_text']]
# val = train[3900:]
# train = train[:3900]
#
# train.to_csv('data/train.txt', sep='\t', header=False, index=False)
# val.to_csv('data/val.txt', sep='\t', header=False, index=False)
# test.to_csv('data/test.txt', sep='\t', header=False, index=False)

with open('output/test_results.tsv') as f:
    re = pd.read_csv('data/contest_results.csv')
    re.flag = [i[-2] for i in f]
    print(re)
    re.to_csv('data/contest_results.csv', index=False)