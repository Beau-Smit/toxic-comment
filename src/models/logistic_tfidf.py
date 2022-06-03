import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
# import eli5

dataset = pd.read_csv("../data/sample_data.csv")

train, test = train_test_split(dataset, test_size=0.2, random_state=23)
train_text = train.comment_text.tolist()
test_text = test.comment_text.tolist()
train_labels = train.toxic
test_labels = test.toxic

# preprocess into tf-idf
vec=TfidfVectorizer(stop_words="english", ngram_range=(1, 1), use_idf=True) # use_idf allows idf reweighting
train_tfIdf = vec.fit_transform(train_text)
test_tfIdf = vec.transform(test_text)
# df = pd.DataFrame(tfIdf[0].T.todense(), index=vec.get_feature_names_out(), columns=["TF-IDF"])
# df = df.sort_values('TF-IDF', ascending=False)

logit = LogisticRegression(C=5e1, random_state=23, n_jobs=-1)
logit.fit(train_tfIdf, train_labels)
test["prob_toxic"] = logit.predict_proba(test_tfIdf)[:,1]
test["pred_toxic"] = logit.predict(test_tfIdf)

out_df = test[["id", "comment_text", "toxic", "prob_toxic", "pred_toxic"]]
print((out_df.pred_toxic == out_df.toxic).mean())
print(out_df.loc[out_df.pred_toxic == out_df.toxic, ["comment_text", "toxic"]])
print(confusion_matrix(out_df.pred_toxic, out_df.toxic))
print(out_df.toxic.mean())

# eli5.show_weights(estimator=logit, 
#                   feature_names= list(vec.get_feature_names_out()),
#                  top=(50, 5))