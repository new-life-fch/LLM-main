import json

input_file = "wiki_data/clean_corpus.jsonl"
output_file = "wiki_data/flashrag_corpus.jsonl"

with open(input_file, "r", encoding="utf-8") as fin, \
     open(output_file, "w", encoding="utf-8") as fout:
    for line in fin:
        obj = json.loads(line)
        # 拼接 title + text 到 contents，去掉title中的双引号
        title = obj["title"].strip().strip('"')  # 去掉首尾的双引号
        contents = title + "\n" + obj["text"].strip()
        new_obj = {
            "id": str(obj["id"]),   # id 转为字符串
            "contents": contents
        }
        fout.write(json.dumps(new_obj, ensure_ascii=False) + "\n")

print(f"✅ 转换完成: {output_file}")
