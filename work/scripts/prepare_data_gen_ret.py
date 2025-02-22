"""
    prepare data for genret
    input: 2024-cti-bigmodel-retrieval-data/train.csv
    output: genret/train.json
"""

import sys
import json
import os

if not os.path.exists("/home/aistudio/work/data/"):
    os.makedirs("/home/aistudio/work/data/")

if not os.path.exists("/home/aistudio/work/data"):
    os.makedirs("/home/aistudio/work/data")

train_data_src = "/home/aistudio/work/data/fusai-2024-cti-bigmodel-retrieval-data/train_data.csv"
dev_data_src = "/home/aistudio/work/data/fusai-2024-cti-bigmodel-retrieval-data/dev_data.csv"


train_data_tgt = "/home/aistudio/work/data/fusai_query.json"
dev_data_tgt = "/home/aistudio/work/data/dev.json"


def prepare_json_data(filepath):
    all_json_d = []
    with open(filepath, "r") as rfile:
        for line in rfile.readlines():
            line = line.strip()
            spt = line.split("\t")
            query, adid, click = spt
            d = {}
            d['src'] = query
            d['tgt'] = adid
            d['click'] = float(click)
            json_d = json.dumps(d, ensure_ascii=False)
            all_json_d.append(json_d)
    return all_json_d

# prepare train 
train_json_d = prepare_json_data(train_data_src)
print("load train data done")
with open(train_data_tgt, "w") as wfile:
    for json_d in train_json_d:
        wfile.write(json_d + "\n")
print("write train json done")

# prepare dev
dev_json_d = prepare_json_data(dev_data_src)
print("load dev data done")
with open(dev_data_tgt, "w") as wfile:
    for json_d in dev_json_d:
        wfile.write(json_d + "\n")
print("write dev json done")



