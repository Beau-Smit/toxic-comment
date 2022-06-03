import pandas as pd
from preprocessing import clean_text
import time
from joblib import Parallel, delayed

df = pd.read_csv("../input_data/train.csv")

start = time.time()

# parallel
# Parallel(n_jobs=-1)(delayed(clean_text)(i) for i in df.loc[:1000, "comment_text"])
df["cleaned_text"] = Parallel(n_jobs=-1)(delayed(clean_text)(i) for i in df["comment_text"])
end = time.time()
print(f"execution time: {end - start}")

df.to_csv("../input_data/train_clean.csv", index=False)
