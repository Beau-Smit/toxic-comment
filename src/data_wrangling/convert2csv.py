import pandas as pd
import os
import json

root = f"../raw_data/Wulczyn2017toxic"
for fname in os.listdir(f"{root}"):
    name, ext = os.path.splitext(fname)
    if ext == ".tsv":
        df = pd.read_table(f"{root}/{fname}", sep="\t")
        df.to_csv(f"{root}/{name}.csv")
    elif ext == ".json":
        out_lst = []
        df = pd.read_json(f"../raw_data/{fname}")
        for item in df.conan:
            record = json.loads(json.dumps(item))
            out_lst.append(record)
        out_df = pd.DataFrame(out_lst)
        out_df.to_csv(f"../raw_data/{name}.csv")
