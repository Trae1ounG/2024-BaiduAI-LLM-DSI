from ollama import Client
client = Client(host='http://localhost:11435')
import json
from tqdm import tqdm
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

def gen_response(passage):
    response = client.chat(model='qwen2:7b', messages=[
        {
            'role': 'user',
            'content': f'你是一个网页搜索请求生成专家，将给定你一组以;分割的输入。输入内容是网页实际内容，请你选择最能代表网页内容的词生成搜索请求，要求词必须明确指向该网页而不能模糊。输出内容用;分割。\n示例：\n输入：VR线上看房;达人探盘;幸福里;房屋估价;二手房;找房新平台;南北通透;小区品质;绿化率高\n输出：哪里可以VR线上看房;幸福里APP;找房子新平台;哪里可以找小区\n本次输入：{passage}\n本次输出：',
        },
    ])
    return response['message']['content']

def process_line(data, output_file):
    start_time = time.time()
    tgt, src = data.split("\t")
    src = ";".join(src.split("<block>"))
    # print(f"输入:{src}")
    res = gen_response(src)
    # print(f"输出:{res}")
    result = {"tgt": tgt, "src": res}
    end_time = time.time()
    print(f"cost_time:{end_time-start_time:.2f}")
    with open(output_file, 'a') as w:
        w.write(json.dumps(result, ensure_ascii=False) + "\n")
    return result

def main():
    input_file = '/home/aistudio/data/data274453/fusai-2024-cti-bigmodel-retrieval-data/ad_landingpage.txt'
    output_file = '/home/aistudio/work/scripts/ollama/train_indexing_qwen_merge.json'

    # start = 0
    # end = 444214+70000
    with open(input_file, 'r') as f:
        lines = f.readlines()
    # 读取output文件中的tgt，并转换为int，存入集合1
    with open(output_file, 'r') as f:
        tgt_set = {int(json.loads(line)['tgt'])-1 for line in f}
    # 读取input文件的指定范围的行
    with open(input_file, 'r') as f:
        input_lines = f.readlines()
    # 计算新集合，input行号集合减去集合1
    new_indices = set(range(start,end)) - tgt_set
    # 从input文件读取新集合对应的行
    new_lines = [input_lines[i - start] for i in new_indices]

    with ThreadPoolExecutor(max_workers=20) as executor:
        future_to_data = {executor.submit(process_line, data, output_file): data for data in new_lines}
        for future in tqdm(as_completed(future_to_data), total=len(new_lines)):
            try:
                future.result()
            except Exception as exc:
                print(f'generated an exception: {exc}')

if __name__ == "__main__":
    main()