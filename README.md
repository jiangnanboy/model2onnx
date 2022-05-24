一.将roberta模型转为onnx，并对[MASK]进行推理预测。

二.将macbert模型转为onnx，并对中文进行拼写纠错。

#### 一.roberta-onnx（对中文句子[MASK]预测）
##### 将模型转为onnx
模型在从[这里](https://huggingface.co/uer/chinese_roberta_L-2_H-512) 下载。
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

#### 二.macbert-onnx（中文拼写纠错）
##### 将模型转为onnx
模型在从[这里](https://huggingface.co/shibing624/macbert4csc-base-chinese) 下载。
```
def load_model():
    tokenizer = BertTokenizer.from_pretrained("../model/macbert4csc-base-chinese")
    model = BertForMaskedLM.from_pretrained("../model/macbert4csc-base-chinese")
    return model, tokenizer

def convert2onnx(model, tokenizer, save_path):
    convert('pt', model, Path(save_path), 11, tokenizer)
```
##### 测试用例
```
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

结果：
    你找到你最喜欢的工作，我也很高心。  =>  你找到你最喜欢的工作，我也很高兴。 [('心', '兴', 15, 16)]
```