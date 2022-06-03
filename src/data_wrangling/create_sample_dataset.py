import pandas as pd
import os

jigsaw_dir = f"../raw_data/jigsaw-toxic-comment-classification-challenge"
train_path = os.path.join(jigsaw_dir, "train.csv")
train_df = pd.read_csv(train_path)
sample_train_df = train_df.sample(n=10000)

out_path = os.path.join("../data", "sample_data.csv")
sample_train_df.to_csv(out_path, index=False)
