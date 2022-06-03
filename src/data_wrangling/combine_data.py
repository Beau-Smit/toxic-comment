import pandas as pd
from preprocessing import clean_text
from joblib import Parallel, delayed
import time, re

# get and clean the new twitter data
data_path = "C:/Users/Beau/Documents/UChicago/Spring2022/Adv_ML/toxic-comments/raw_data/davidson2017.csv"

new_data = pd.read_csv(data_path)

new_data.loc[new_data["class"] != 2, "label"] = 1
new_data.loc[new_data["class"] == 2, "label"] = 0

new_data["comment_text"] = new_data["tweet"]

def clean(x):
    x = re.sub("[!@]", "", x)
    return re.sub("^ RT .+:", "", x)
new_data["comment_text"] = new_data["comment_text"].apply(clean)
new_data["cleaned_text"] = Parallel(n_jobs=-1)(delayed(clean_text)(i) for i in new_data["comment_text"])

# get the current training data
train_path = "C:/Users/Beau/Documents/UChicago/Spring2022/Adv_ML/toxic-comments/input_data/train_clean.csv"

train = pd.read_csv(train_path)
train['label'] =  (train[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']]\
                     .sum(axis=1) > 0).astype(int)
train = train[["comment_text", "cleaned_text", "label"]]

# combine the datasets
train_combined = pd.concat([train, new_data])
train_combined.to_csv("C:/Users/Beau/Documents/UChicago/Spring2022/Adv_ML/toxic-comments/input_data/train_clean_4.csv", index=False)
