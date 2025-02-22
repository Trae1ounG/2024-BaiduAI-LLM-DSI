import ollama
import json
from tqdm import *
import time
import os
import argparse
from ollama import Client
def parse_args():
    """
        parse args
    """
    parser = argparse.ArgumentParser(__doc__)
    # parser.add_argument('--host', type=str)
    args = parser.parse_args()
    return args
print(f"os:{os.getpid()}")
args = parse_args()

def gen_response(src,input):
    if len(input) > 6:
        input = input[:6]
    input = ";".join(input)
    print(f"输入:\n{input}")
    if src == None:
        response = ollama.chat(model='qwen2', messages=[
        {
            'role': 'user',
            'content': f'你是一个网页搜索请求生成专家，将给定你一组已有的相关搜索词，要求你根据相关搜索词，输出三条可能被其他用户使用的但不包含在输入中的搜索词，长度在6到20之间。要求返回内容只包含生成搜索词，并且包含最重要的关键信息，输出使用;分割\n示例：\n输入：\n相关搜索词:上门提供沙发翻新解决方案;苏州地区专业沙发维修保养;沙发翻新换皮\n输出：沙发换新服务;苏州沙发翻新;沙发上门翻新服务\n本次输入：\n相关搜索词:{input}\n本次输出：',
                },
        ])
    else:
        response = ollama.chat(model='qwen2', messages=[
        {
            'role': 'user',
            'content': f'你是一个网页搜索请求生成专家，将给定你一组已有的相关搜索词和最近一次搜索词，要求你根据相关搜索词，输出三条可能被其他用户使用的但不包含在输入中的与最近一次搜索词最接近的搜索词，长度在6到20之间。要求返回内容只包含生成搜索词，并且包含最重要的关键信息，输出使用;分割\n示例：\n输入：\n相关搜索词:上门提供沙发翻新解决方案;苏州地区专业沙发维修保养;沙发翻新换皮\n最近一次搜索词:沙发维修保养服务\n输出：沙发换新服务;苏州沙发翻新;沙发上门翻新服务\n本次输入：\n相关搜索词:{input}\n最近搜索词:{src}\n本次输出：',
                },
        ])
    return response['message']['content']

def get_tgts():
    d_max_click = dict()
    d_query = dict()
    with open('/root/data/fusai_train_data.json', 'r') as w:
        for line in w:
            data = json.loads(line)
            src, tgt, click = data['src'], int(data['tgt']), float(data['click'])
            if tgt not in d_query:
                d_query[tgt] = []
            d_query[tgt].append([src,click])
            if tgt not in d_max_click:
                d_max_click[tgt] = [click, src]
            else:
                if click > d_max_click[tgt][0]:
                    d_max_click[tgt] = [click, src]
    # click的分位数 (25%, 50%, 90%): [1. 1. 4.]
    # 每个tgt包含的[src, click]对个数的分位数 (10%, 50%, 75%): [4.4 6.  8.5]
    use_tgts = []
    for tgt in range(1,888427 + 1):
        if tgt not in d_query or (d_max_click[tgt][0] >= 4.0 or len(d_query) < 5):
            use_tgts.append(tgt)
            continue
    print("=====处理完毕=====")
    return d_max_click,use_tgts

d_query = dict()
d_query_back = dict()
with open('/home/aistudio/data/data274453/fusai-2024-cti-bigmodel-retrieval-data/fusai_train_data.json', 'r') as w:
    for line in w:
        data = json.loads(line)
        src, tgt, click  = data['src'], int(data['tgt']), float(data['click'])
        if tgt not in d_query:
            d_query[tgt] = []
        d_query[tgt].append([src,click])
with open('/home/aistudio/work/data/fusai_core_terms_to_query.json', 'r') as w:
    for line in w:
        data = json.loads(line)
        src, tgt, click  = data['src'], int(data['tgt']), 0
        if tgt not in d_query:
            d_query[tgt] = []
        d_query[tgt].append([src,click])  
with open('/home/aistudio/work/data/fusai_cleaned_ollama_1.json', 'r') as w:
    for line in w:
        data = json.loads(line)
        src, tgt, click  = data['src'], int(data['tgt']), 0
        if tgt not in d_query:
            d_query[tgt] = []
        d_query[tgt].append([src, click]) 

d_max_click, keys = get_tgts()
with open('/home/aistudio/work/scripts/ollama/test_ollama_group_data.json', 'w', buffering=1024*1024) as w1:
    for tgt in tqdm(keys):
        start_time = time.time()
        srcs = d_query[tgt]
        print(f"当前tgt_id:{tgt}")
        srcs_ = [src for src, click in srcs]
        # if len(srcs) > 5:
            # srcs = srcs[:5]
        # for src, click in srcs:
            # if click >= 4.0:
        if tgt in d_max_click:
            src = d_max_click[tgt][1]
        else:
            src = None
        res = gen_response(src, srcs_)
        print(f"输出:\n{res}")
        print("============")
        w1.write(json.dumps({"tgt":tgt, "src":res},ensure_ascii=False) + "\n")
        end_time = time.time()
        print(f"cost_time:{end_time-start_time:.2f}")