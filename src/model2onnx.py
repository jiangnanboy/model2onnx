from transformers import BertTokenizer, BertForMaskedLM
import numpy as np
from pathlib import Path
from transformers.convert_graph_to_onnx import convert
import onnx
import onnxruntime as ort

def load_model():
    tokenizer = BertTokenizer.from_pretrained("../model/chinese_roberta_L-2_H-512")
    model = BertForMaskedLM.from_pretrained("../model/chinese_roberta_L-2_H-512")
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
    # model, tokenizer = load_model()
    # convert2onnx(model, tokenizer, '../model/onnx/roberta.onnx')
    onnx_test('../model/onnx/roberta.onnx')


