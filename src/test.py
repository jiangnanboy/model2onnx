from transformers import BertTokenizer, BertForMaskedLM
from transformers import pipeline
import torch
import numpy as np
from pathlib import Path
from transformers.convert_graph_to_onnx import convert
import onnx
import onnxruntime as ort

def test1():
    unmasker = pipeline('fill-mask', model='../model/chinese_roberta_L-2_H-512')
    print(unmasker("中国的首都是[MASK]京。"))

def test2():
    tokenizer = BertTokenizer.from_pretrained("../model/chinese_roberta_L-2_H-512")
    model = BertForMaskedLM.from_pretrained("../model/chinese_roberta_L-2_H-512")
    fill_mask = pipeline(
        "fill-mask",
        model=model,
        tokenizer=tokenizer
    )
    print(fill_mask(f"中国的首都是{fill_mask.tokenizer.mask_token}京。"))

def test3():
    tokenizer = BertTokenizer.from_pretrained("../model/chinese_roberta_L-2_H-512")
    model = BertForMaskedLM.from_pretrained("../model/chinese_roberta_L-2_H-512")
    sent = '中国的首都是' + tokenizer.mask_token + '京。'
    tokens = ['[CLS]'] + tokenizer.tokenize(sent) + ['[SEP]']
    masked_ids = torch.tensor([tokenizer.convert_tokens_to_ids(tokens)])
    segments_ids = torch.tensor([[0] * len(tokens)])
    outputs = model(masked_ids, token_type_ids=segments_ids)
    predictions_scores = outputs[0]
    print(tokens)
    print('mask: {} -> mask_id: {}'.format(tokenizer.mask_token, tokenizer.mask_token_id))
    predictions_index = torch.argmax(predictions_scores[0, tokens.index('[MASK]')]).item()
    predictions_token = tokenizer.convert_ids_to_tokens([predictions_index])[0]
    print(predictions_token)

def test4():
    tokenizer = BertTokenizer.from_pretrained("../model/chinese_roberta_L-2_H-512")
    model = BertForMaskedLM.from_pretrained("../model/chinese_roberta_L-2_H-512")
    sent = '中国的首都是' + tokenizer.mask_token + '京。'
    tokenized_tokens =  tokenizer(sent)
    mask_idx = tokenized_tokens["input_ids"].index(tokenizer.convert_tokens_to_ids("[MASK]"))
    input_ids = np.array([tokenized_tokens['input_ids']])
    attention_mask = np.array([tokenized_tokens['attention_mask']])
    token_type_ids = np.array([tokenized_tokens['token_type_ids']])

    model.eval()
    with torch.no_grad():
        predictions = model(input_ids=torch.LongTensor(input_ids),
                            attention_mask=torch.LongTensor(attention_mask),
                            token_type_ids=torch.LongTensor(token_type_ids))
        predicted_index = torch.argmax(predictions[0][0][mask_idx])
        predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
        print(predicted_index)
        print(predicted_token)

    return model, tokenizer

def convert2onnx(model, tokenizer, save_path):
    convert('pt', model, Path(save_path), 11, tokenizer)

def onnx_test(onnx_path, topk=5):
    tokenizer = BertTokenizer.from_pretrained("../model/chinese_roberta_L-2_H-512")
    sent = '中国的首都是' + tokenizer.mask_token + '京。'
    tokenized_tokens = tokenizer(sent)
    mask_idx = tokenized_tokens["input_ids"].index(tokenizer.convert_tokens_to_ids("[MASK]"))
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
    predicted_index = np.argsort(-result[0][mask_idx])[:topk]
    for index in predicted_index:
        predicted_token = tokenizer.convert_ids_to_tokens([index])[0]
        print('index : {} -> token : {}'.format(index, predicted_token))


if __name__ == '__main__':
    model, tokenizer = test4()
    convert2onnx(model, tokenizer, '../model/onnx/roberta.onnx')
    # onnx_test('../model/onnx/roberta.onnx')
    test2()

