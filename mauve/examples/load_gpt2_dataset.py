import json
from pathlib import Path


def load_gpt2_dataset(json_file_name, num_examples=float('inf')):
    texts = []
    for i, line in enumerate(open(json_file_name)):
        if i >= num_examples:
            break
        texts.append(json.loads(line)['text'])
    return texts


def load_json_dataset(json_file_name, num_examples=float('inf')):
    output_texts = []
    if Path(json_file_name).suffix == ".json":
        with open(json_file_name) as f:
            texts = json.load(f)
        for i, text in enumerate(texts):
            if i >= num_examples:
                break
            output_texts.append(text)
    elif Path(json_file_name).suffix == ".jsonl":
        for i, line in enumerate(open(json_file_name)):
            if i >= num_examples:
                break
            output_texts.append(json.loads(line))
    return output_texts
