from transformers import BertTokenizer, BertForMaskedLM
import numpy as np
from pathlib import Path
from transformers.convert_graph_to_onnx import convert
import onnx
import onnxruntime as ort
import operator
import torch

def load_model():
    tokenizer = BertTokenizer.from_pretrained("../model/macbert4csc-base-chinese")
    model = BertForMaskedLM.from_pretrained("../model/macbert4csc-base-chinese")
    return model, tokenizer

def convert2onnx(model, tokenizer, save_path):
    convert('pt', model, Path(save_path), 11, tokenizer)

def get_errors(corrected_text, origin_text):
    sub_details = []
    for i, ori_char in enumerate(origin_text):
        if ori_char in [' ', '“', '”', '‘', '’', '琊', '\n', '…', '—', '擤']:
            # add unk word
            corrected_text = corrected_text[:i] + ori_char + corrected_text[i:]
            continue
        if i >= len(corrected_text):
            continue
        if ori_char != corrected_text[i]:
            if ori_char.lower() == corrected_text[i]:
                # pass english upper char
                corrected_text = corrected_text[:i] + ori_char + corrected_text[i + 1:]
                continue
            sub_details.append((ori_char, corrected_text[i], i, i + 1))
    sub_details = sorted(sub_details, key=operator.itemgetter(2))
    return corrected_text, sub_details

def onnx_test(onnx_path):
    tokenizer = BertTokenizer.from_pretrained("../model/macbert4csc-base-chinese")
    sent = '你找到你最喜欢的工作，我也很高心。'
    tokenized_tokens = tokenizer(sent)
    input_ids = np.array([tokenized_tokens['input_ids']], dtype=np.int64)
    attention_mask = np.array([tokenized_tokens['attention_mask']], dtype=np.int64)
    token_type_ids = np.array([tokenized_tokens['token_type_ids']], dtype=np.int64)

    model = onnx.load(onnx_path)
    sess = ort.InferenceSession(bytes(model.SerializeToString()))
    result = sess.run(
        output_names=None,
        input_feed={"input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "token_type_ids": token_type_ids}
    )[0]
    _text = tokenizer.decode(np.argmax(result[0], axis=-1), skip_special_tokens=True).replace(' ', '')
    print('_text: {}'.format(_text))
    corrected_text = _text[:len(sent)]
    corrected_text, details = get_errors(corrected_text, sent)
    print(sent, ' => ', corrected_text, details)

if __name__ == '__main__':
    # model, tokenizer = load_model()
    # convert2onnx(model, tokenizer, '../model/macbert4csc_onnx/macbert4csc.onnx')
    onnx_test('../model/macbert4csc_onnx/macbert4csc.onnx')


