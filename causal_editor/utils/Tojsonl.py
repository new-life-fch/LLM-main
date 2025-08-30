import argparse
from tqdm import tqdm
import re
import html
import os
import json
import subprocess
from pathlib import Path
import shutil
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool


def load_corpus(dir_path):
    def iter_files(path):
        """Walk through all files located under a root path."""
        if os.path.isfile(path):
            yield path
        elif os.path.isdir(path):
            for dirpath, _, filenames in os.walk(path):
                for f in filenames:
                    yield os.path.join(dirpath, f)
        else:
            raise RuntimeError("Path %s is invalid" % path)

    def read_jsonl_file(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                json_data = json.loads(line)
                corpus.append(json_data)

    all_files = [file for file in iter_files(dir_path)]
    corpus = []

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        for file_path in all_files:
            executor.submit(read_jsonl_file, file_path)

    return corpus


def basic_process(title, text):
    title = html.unescape(title)
    text = html.unescape(text)
    text = text.strip()

    if "(disambiguation)" in title.lower():
        return None, None
    if "(disambiguation page)" in title.lower():
        return None, None
    # Take out List/Index/Outline pages (mostly links)
    if re.match(r"(List of .+)|(Index of .+)|(Outline of .+)", title):
        return None, None
    if text.startswith("REDIRECT") or text.startswith("redirect"):
        return None, None
    if text.endswith(". References."):
        text = text[: -len(" References.")].strip()

    # ========= 正则清理部分 (basic_process里) =========
    text = re.sub(r"\{\{cite .*?\}\}", " ", text, flags=re.DOTALL)
    text = re.sub(r"\| ?item[0-9]?_?style= ?.*? ", " ", text)
    text = re.sub(r"\| ?col[0-9]?_?style= ?.*? ", " ", text)
    text = re.sub(r"\| ?row[0-9]?_?style= ?.*? ", " ", text)
    text = re.sub(r"\| ?style= ?.*? ", " ", text)
    text = re.sub(r"\| ?bodystyle= ?.*? ", " ", text)
    text = re.sub(r"\| ?frame_?style= ?.*? ", " ", text)
    text = re.sub(r"\| ?data_?style= ?.*? ", " ", text)
    text = re.sub(r"\| ?label_?style= ?.*? ", " ", text)
    text = re.sub(r"\| ?headerstyle= ?.*? ", " ", text)
    text = re.sub(r"\| ?list_?style= ?.*? ", " ", text)
    text = re.sub(r"\| ?title_?style= ?.*? ", " ", text)
    text = re.sub(r"\| ?ul_?style= ?.*? ", " ", text)
    text = re.sub(r"\| ?li_?style= ?.*? ", " ", text)
    text = re.sub(r"\| ?border-style= ?.*? ", " ", text)
    text = re.sub(r'\|? ?style=".*?"', "", text)
    text = re.sub(r'\|? ?rowspan=".*?"', "", text)
    text = re.sub(r'\|? ?colspan=".*?"', "", text)
    text = re.sub(r'\|? ?scope=".*?"', "", text)
    text = re.sub(r'\|? ?align=".*?"', "", text)
    text = re.sub(r'\|? ?valign=".*?"', "", text)
    text = re.sub(r'\|? ?lang=".*?"', "", text)
    text = re.sub(r'\|? ?bgcolor=".*?"', "", text)
    text = re.sub(r"\|? ?bg=\#[a-z]+", "", text)
    text = re.sub(r'\|? ?width=".*?"', "", text)
    text = re.sub(r"\|? ?height=[0-9]+", "", text)
    text = re.sub(r"\|? ?width=[0-9]+", "", text)
    text = re.sub(r"\|? ?rowspan=[0-9]+", "", text)
    text = re.sub(r"\|? ?colspan=[0-9]+", "", text)
    text = re.sub(r"\|? ?align=[a-z]+", "", text)
    text = re.sub(r"\|? ?valign=[a-z]+", "", text)
    text = re.sub(r"\|? ?scope=[a-z]+", "", text)
    text = re.sub(r"File:[A-Za-z0-9 ]+\.[a-z]{3,4}(\|[0-9]+px)?", "", text)
    text = re.sub(r"Source: \[.*?\]", "", text)

    text = text.replace("Country flag|", "country:")
    text = text.replace("flag|", "country:")
    text = text.replace("flagicon|", "country:")
    text = text.replace("flagcountry|", "country:")
    text = text.replace("Flagu|", "country:")
    text = text.replace("display=inline", "")
    text = text.replace("display=it", "")
    text = text.replace("abbr=on", "")
    text = text.replace("disp=table", "")

    title = title.replace("\n", " ").replace("\t", " ")

    return title, text


def split_list(lst, n):
    """Split a list into n roughly equal parts."""
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n)]


def single_worker(docs):
    results = []
    for item in tqdm(docs):
        title, text = basic_process(item[0], item[1])
        if title is None:
            continue
        title = f'"{title}"'
        results.append((title, text))
    return results


def process_batch_chunking(all_title, all_text, batch_clean_corpus, args, global_idx):
    """处理单个批次的分块操作"""
    print("Start chunking batch...")
    idx = 0
    
    if args.use_chonkie:
        print("Using Chonkie chunker...")
        # Initialize a Chonkie chunker, based on the chunk_by argument
        if args.chunk_by == "token":
            chunker = chonkie.TokenChunker(tokenizer=args.tokenizer_name_or_path, chunk_size=args.chunk_size)
        elif args.chunk_by == "sentence":
            chunker = chonkie.SentenceChunker(tokenizer_or_token_counter=args.tokenizer_name_or_path, chunk_size=args.chunk_size)
        elif args.chunk_by == "recursive":
            chunker = chonkie.RecursiveChunker(
                tokenizer_or_token_counter=args.tokenizer_name_or_path, chunk_size=args.chunk_size, min_characters_per_chunk=1
            )
        elif args.chunk_by == "100w":
            chunker = chonkie.TokenChunker(tokenizer="word", chunk_size=100)
        else:
            raise ValueError(f"Invalid chunking method: {args.chunk_by}")

        # Chunk the text into segments, with chunker
        for title, text in tqdm(zip(all_title, all_text), total=len(all_text), desc="Chunking with Chonkie"):
            chunks = chunker.chunk(text)
            for chunk in chunks:
                item = {"id": global_idx, "title": title, "text": chunk.text}
                batch_clean_corpus.append(item)
                global_idx += 1
    else:
        print("Using default chunker...")
        if args.chunk_by == "sentence":
            for doc in tqdm(nlp.pipe(all_text, n_process=args.num_workers, batch_size=2000), total=len(all_text), desc="Chunking by sentence"):
                title = all_title[idx]
                idx += 1
                sentences = [sent.text.strip() for sent in doc.sents]
                segments = []
                for i in range(0, len(sentences), args.stride):
                    segment = " ".join(sentences[i : i + args.seg_size])
                    segments.append(segment)
                    if i + args.seg_size >= len(sentences):
                        break
                for segment in segments:
                     text = segment.replace("\n", " ").replace("\t", " ")
                     item = {"id": global_idx, "title": title, "text": text}
                     batch_clean_corpus.append(item)
                     global_idx += 1

        elif args.chunk_by == "100w":
            for doc in tqdm(nlp.pipe(all_text, n_process=args.num_workers, batch_size=2000), total=len(all_text), desc="Chunking by 100 words"):
                title = all_title[idx]
                idx += 1
                segments = []
                word_count = 0
                segment_tokens = []
                for token in doc:
                    segment_tokens.append(token.text_with_ws)
                    if not token.is_space and not token.is_punct:
                        word_count += 1
                        if word_count == 100:
                            word_count = 0
                            segments.append("".join([token for token in segment_tokens]))
                            segment_tokens = []
                if word_count != 0:
                    for token in doc:
                        segment_tokens.append(token.text_with_ws)
                        if not token.is_space and not token.is_punct:
                            word_count += 1
                            if word_count == 100:
                                word_count = 0
                                segments.append("".join([token for token in segment_tokens]))
                                break
                if word_count != 0:
                    segments.append("".join([token for token in segment_tokens]))

                for segment in segments:
                     text = segment.replace("\n", " ").replace("\t", " ")
                     item = {"id": global_idx, "title": title, "text": text}
                     batch_clean_corpus.append(item)
                     global_idx += 1
    
    return global_idx


if __name__ == "__main__":
    # 设置缓存目录到wiki_data/cache
    import os
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "wiki_data", "cache")
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["HF_DATASETS_CACHE"] = cache_dir
    os.environ["TRANSFORMERS_CACHE"] = cache_dir
    
    parser = argparse.ArgumentParser(description="Generate clean wiki corpus file for indexing.")
    parser.add_argument("--dump_path", type=str)
    parser.add_argument(
        "--use_chonkie",
        action="store_true",
        help="Use Chonkie for chunking (default: False)"
    )


    parser.add_argument("--chunk_by", default="token", choices=["token", "sentence", "recursive", "100w"], type=str)
    parser.add_argument("--chunk_size", default=512, type=int)
    parser.add_argument("--tokenizer_name_or_path", default="o200k_base", type=str)
    parser.add_argument("--seg_size", default=None, type=int)
    parser.add_argument("--stride", default=None, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--save_path", type=str, default="clean_corpus.jsonl")
    args = parser.parse_args()

    if args.use_chonkie:
        import chonkie
    else:
        assert args.chunk_by in ["100w", "sentence"], "Only supports sentence and 100w chunking without chonkie!"
        import spacy

        nlp = spacy.load("en_core_web_lg")

    from datasets import load_dataset
    import gc

    print("Loading .arrow dataset...")
    dataset = load_dataset("arrow", data_files=args.dump_path, split="train")
    
    # 设置批处理大小
    batch_size = 300000  # 每批处理300000条数据
    total_samples = len(dataset)
    print(f"Total samples: {total_samples}, processing in batches of {batch_size}")
    
    # 初始化输出文件
    output_file = open(args.save_path, "w", encoding="utf-8")
    global_idx = 0
    
    # 分批处理数据
    for batch_start in range(0, total_samples, batch_size):
        batch_end = min(batch_start + batch_size, total_samples)
        print(f"Processing batch {batch_start//batch_size + 1}/{(total_samples-1)//batch_size + 1}: samples {batch_start}-{batch_end-1}")
        
        # 获取当前批次数据
        batch_dataset = dataset.select(range(batch_start, batch_end))
        
        corpus = []
        for item in batch_dataset:
            corpus.append({"title": item["title"], "text": item["text"]})
        
        documents = {}
        # To avoid duplicate pages within this batch
        for item in tqdm(corpus, desc="Merging duplicates"):
            title = item["title"]
            text = item["text"]
            if title in documents:
                documents[title] += " " + text
            else:
                documents[title] = text
        
        print("Start pre-processing batch...")
        documents = list(documents.items())
        
        with Pool(processes=args.num_workers) as p:
            result_list = list(tqdm(p.imap(single_worker, split_list(documents, args.num_workers)), desc="Pre-processing"))
        result_list = sum(result_list, [])
        
        all_title = [item[0] for item in result_list]
        all_text = [item[1] for item in result_list]
        
        # 处理当前批次的分块
        batch_clean_corpus = []
        global_idx = process_batch_chunking(all_title, all_text, batch_clean_corpus, args, global_idx)
        
        # 写入当前批次结果
        for item in batch_clean_corpus:
            output_file.write(json.dumps(item) + "\n")
        
        # 清理内存
        del corpus, documents, result_list, all_title, all_text, batch_clean_corpus, batch_dataset
        gc.collect()
        
        print(f"Batch {batch_start//batch_size + 1} completed, {global_idx} total chunks processed so far.")
    
    output_file.close()
    print("Finish!")
    exit(0)  # 提前退出，避免执行后面的代码
