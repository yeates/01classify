from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import pandas as pd
import jieba
import xgboost as xgb

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
train.title.fillna(value='')
train.content.fillna(value='')
test.title.fillna(value='')
test.content.fillna(value='')

train.title = [' '.join(jieba.cut(str(title))) for title in train.title]
train.content = [' '.join(jieba.cut(str(content))) for content in train.content]
test.title = [' '.join(jieba.cut(str(title))) for title in test.title]
test.content = [' '.join(jieba.cut(str(content))) for content in test.content]

vectorizer = TfidfVectorizer(binary=False, decode_error='ignore')
train_features = vectorizer.fit_transform([str(a)+str(b) for a, b in zip(train.title, train.content)])
print("Train Feature Size: " + str(train_features.shape))

model = SVC(C=1.0, kernel="linear")

#model = xgb.XGBClassifier(learning_rate=0.1, booster='gbtree', n_estimators=1000,
                         # max_depth=4, min_child_weight=1,
                         # gamma=0, subsample=0.8, colsample_bytree=0.8,
                         # objective='binary:logistic',scale_pos_weight=1, seed=10)

model.fit(train_features, train.flag)

test_features = vectorizer.transform([str(a)+str(b) for a, b in zip(test.title, test.content)])
pred = model.predict(test_features)

output_csv = pd.DataFrame({
   'id': test.id,
   'flag': list(pred)
   })[['id', 'flag']]
output_csv.to_csv('baseline.csv', index=False)

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')



