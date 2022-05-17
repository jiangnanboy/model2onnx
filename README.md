一个实例，将chinese_robera模型转为onnx，将利用onnx进行推理。

##### 将模型转为onnx
pytorch模型在从[这里](https://huggingface.co/uer/chinese_roberta_L-2_H-512) 下载。
```
def load_model():
    tokenizer = BertTokenizer.from_pretrained("../model/chinese_roberta_L-2_H-512")
    model = BertForMaskedLM.from_pretrained("../model/chinese_roberta_L-2_H-512")
    return model, tokenizer

def convert2onnx(model, tokenizer, save_path):
    convert('pt', model, Path(save_path), 11, tokenizer)
```
##### 测试用例
```
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

结果：
    index : 1266 -> token : 北
    index : 1298 -> token : 南
    index : 691 -> token : 东
    index : 4242 -> token : 燕
    index : 3307 -> token : 望
```