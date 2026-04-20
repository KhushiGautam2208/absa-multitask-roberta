import xml.etree.ElementTree as ET
from collections import Counter
from transformers import RobertaTokenizer
from tqdm import tqdm
import torch
import json
import os
import random
from utils.helpers import set_seed

def parse_with_sentiment(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    examples = []
    for sentence in root.findall('sentence'):
        text_elem = sentence.find('text')
        if text_elem is None or text_elem.text is None:
            continue
        text = text_elem.text
        aspects = []
        aspects_elem = sentence.find('aspectTerms')
        if aspects_elem is not None:
            for term in aspects_elem.findall('aspectTerm'):
                term_text = term.get('term')
                polarity = term.get('polarity')
                implicit = (term.get('implicit_sentiment') == "True")
                aspects.append({'term': term_text, 'polarity': polarity, 'implicit': implicit})
        if not aspects:
            continue
        pols = [a['polarity'] for a in aspects]
        cnt = Counter(pols)
        if cnt['positive'] >= cnt['negative'] and cnt['positive'] >= cnt['neutral']:
            sent_label = 'positive'
        elif cnt['negative'] >= cnt['positive'] and cnt['negative'] >= cnt['neutral']:
            sent_label = 'negative'
        else:
            sent_label = 'neutral'
        sent_id = {'positive':0, 'neutral':1, 'negative':2}[sent_label]
        examples.append({'text': text, 'aspects': aspects, 'sentiment_id': sent_id})
    return examples

def tokenize_and_label(examples, tokenizer, max_len=None):
    processed = []
    for ex in tqdm(examples, desc="Tokenizing"):
        text = ex['text']
        tokens = tokenizer.tokenize(text)
        labels = ['O'] * len(tokens)
        encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        offsets = encoding['offset_mapping']
        if len(offsets) != len(tokens):
            continue
        for aspect in ex['aspects']:
            term = aspect['term']
            start_char = text.find(term)
            if start_char == -1:
                continue
            end_char = start_char + len(term)
            token_start = None
            token_end = None
            for i, (off_start, off_end) in enumerate(offsets):
                if off_start >= end_char:
                    break
                if off_end <= start_char:
                    continue
                if token_start is None:
                    token_start = i
                token_end = i
            if token_start is None:
                continue
            labels[token_start] = 'B-ASPECT'
            for i in range(token_start+1, token_end+1):
                labels[i] = 'I-ASPECT'
        has_implicit = any(a['implicit'] for a in ex['aspects'])
        processed.append({
            'tokens': tokens,
            'labels': labels,
            'has_implicit': has_implicit,
            'sentiment_id': ex['sentiment_id']
        })
    if max_len is None:
        max_len = max(len(item['tokens']) for item in processed)
    return processed, max_len

def encode(processed, tokenizer, max_len):
    input_ids_list, att_mask_list, label_ids_list, sent_list, imp_list = [], [], [], [], []
    for item in processed:
        tokens = item['tokens']
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        if len(input_ids) < max_len:
            att_mask = [1]*len(input_ids) + [0]*(max_len - len(input_ids))
            input_ids = input_ids + [tokenizer.pad_token_id]*(max_len - len(input_ids))
        else:
            input_ids = input_ids[:max_len]
            att_mask = [1]*max_len
        label_ids = [{'O':0,'B-ASPECT':1,'I-ASPECT':2}[l] for l in item['labels']]
        if len(label_ids) < max_len:
            label_ids = label_ids + [0]*(max_len - len(label_ids))
        else:
            label_ids = label_ids[:max_len]
        input_ids_list.append(input_ids)
        att_mask_list.append(att_mask)
        label_ids_list.append(label_ids)
        sent_list.append(item['sentiment_id'])
        imp_list.append(1 if item['has_implicit'] else 0)
    return (torch.tensor(input_ids_list, dtype=torch.long),
            torch.tensor(att_mask_list, dtype=torch.long),
            torch.tensor(label_ids_list, dtype=torch.long),
            torch.tensor(sent_list, dtype=torch.long),
            torch.tensor(imp_list, dtype=torch.long))

def create_splits(all_data, seed=42, train_ratio=0.8, val_ratio=0.1):
    set_seed(seed)
    indices = list(range(len(all_data)))
    random.shuffle(indices)
    n_train = int(train_ratio * len(all_data))
    n_val = int(val_ratio * len(all_data))
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train+n_val]
    test_idx = indices[n_train+n_val:]
    return train_idx, val_idx, test_idx

def preprocess_pipeline(raw_xml_paths, tokenizer_name='roberta-base', save_dir='data/'):
    tokenizer = RobertaTokenizer.from_pretrained(tokenizer_name)
    all_examples = []
    for path in raw_xml_paths:
        all_examples.extend(parse_with_sentiment(path))
    print(f"Total aspect‑annotated sentences: {len(all_examples)}")
    train_idx, val_idx, test_idx = create_splits(all_examples, seed=42)
    train_ex = [all_examples[i] for i in train_idx]
    val_ex   = [all_examples[i] for i in val_idx]
    test_ex  = [all_examples[i] for i in test_idx]
    train_proc, max_len_train = tokenize_and_label(train_ex, tokenizer)
    val_proc, max_len_val = tokenize_and_label(val_ex, tokenizer)
    test_proc, max_len_test = tokenize_and_label(test_ex, tokenizer)
    max_len = max(max_len_train, max_len_val, max_len_test)
    print(f"Max sequence length: {max_len}")
    train_ids, train_mask, train_labels, train_sent, train_imp = encode(train_proc, tokenizer, max_len)
    val_ids, val_mask, val_labels, val_sent, val_imp = encode(val_proc, tokenizer, max_len)
    test_ids, test_mask, test_labels, test_sent, test_imp = encode(test_proc, tokenizer, max_len)
    os.makedirs(save_dir, exist_ok=True)
    torch.save({'input_ids': train_ids, 'attention_mask': train_mask,
                'aspect_labels': train_labels, 'sentiment_labels': train_sent,
                'has_implicit': train_imp}, f'{save_dir}roberta_train.pt')
    torch.save({'input_ids': val_ids, 'attention_mask': val_mask,
                'aspect_labels': val_labels, 'sentiment_labels': val_sent,
                'has_implicit': val_imp}, f'{save_dir}roberta_val.pt')
    torch.save({'input_ids': test_ids, 'attention_mask': test_mask,
                'aspect_labels': test_labels, 'sentiment_labels': test_sent,
                'has_implicit': test_imp}, f'{save_dir}roberta_test.pt')
    label2id = {'O':0, 'B-ASPECT':1, 'I-ASPECT':2}
    id2label = {v:k for k,v in label2id.items()}
    info = {'max_len': max_len, 'label2id': label2id, 'id2label': id2label,
            'num_labels': 3, 'num_sentiment_classes': 3}
    with open(f'{save_dir}roberta_data_info.json', 'w') as f:
        json.dump(info, f)
    print("Preprocessing complete. Files saved in", save_dir)
