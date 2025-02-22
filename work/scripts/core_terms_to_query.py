import json

def generate_json(file_path, output_file):
    with open(file_path, 'r', encoding='utf-8') as f, \
         open(output_file, 'w', encoding='utf-8') as json_file:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split('\t')
                ad_id = parts[0]
                core_terms = parts[1].split('<sep>')
                if core_terms[0] == "none":
                    continue
                for term in core_terms:
                    if term:  # 确保term不为空
                        json_obj = {
                            "src": term.strip(),
                            "tgt": ad_id
                        }
                        # 序列化JSON对象，并确保每个对象只占一行
                        json_file.write(json.dumps(json_obj, ensure_ascii=False) + '\n')

# File path to your ad_core_term.txt
file_path = "/home/aistudio/work/data/fusai-2024-cti-bigmodel-retrieval-data/ad_core_term.txt"
# Output JSON file path
output_file = '/home/aistudio/work/data/fusai_core_terms_to_query.json'

# Generate and write JSON objects to file
generate_json(file_path, output_file)

print(f"JSON objects have been written to {output_file}.")