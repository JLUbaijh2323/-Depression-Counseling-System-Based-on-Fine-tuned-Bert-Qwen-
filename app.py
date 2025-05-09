from flask import Flask, request, render_template_string
import re
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline
)
from peft import PeftModel

# ================ 模型路径配置 ================
MODEL_PATHS = {
    "distilbert": "main/distilbert_final_model",
    "qwen_base": r"C:\Users\baijh2323\项目\项目\认知科学导论期末项目\main\QwenQwen2.5-0.5B-Instruct",
    "lora": r"C:\Users\baijh2323\项目\项目\认知科学导论期末项目\main\Qwen2.5-0.5B-Instruct\lora\train_2025-05-06-18-10-37\checkpoint-180"
}

# ================ 标签映射 ================
ID2TAG = {
    'LABEL_0': ('S2', 1),   # 情绪低落
    'LABEL_1': ('S3', 1),   # 睡眠障碍
    'LABEL_2': ('S9', 1),   # 自杀倾向
    'LABEL_3': ('隐喻', '自然现象隐喻'),
    'LABEL_4': ('隐喻', '机械故障隐喻'),
    'LABEL_5': ('隐喻', '空间压迫隐喻'),
    'LABEL_6': ('抑郁', 1)
}

# ================ 加载模型 ================
def load_models():
    # 文本分类
    cls_model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATHS["distilbert"])
    cls_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATHS["distilbert"])
    # 文本生成
    gen_base = AutoModelForCausalLM.from_pretrained(
        MODEL_PATHS["qwen_base"], device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
    )
    gen_model = PeftModel.from_pretrained(gen_base, MODEL_PATHS["lora"])
    gen_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATHS["qwen_base"])
    return cls_model, cls_tokenizer, gen_model, gen_tokenizer

cls_model, cls_tokenizer, gen_model, gen_tokenizer = load_models()

# ================ 核心函数 ================

def classify_text(text, model, tokenizer, threshold=0.5):
    classifier = pipeline(
        "text-classification", model=model, tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1, top_k=None
    )
    results = classifier(text)[0]
    return [item['label'] for item in results if item['score'] > threshold]


def construct_expert_input(labels):
    symptoms = {'S2': 0, 'S3': 0, 'S9': 0}
    metaphors = []
    depression_flag = 0
    for lbl in labels:
        tag, val = ID2TAG.get(lbl, (None, None))
        if tag in symptoms:
            symptoms[tag] = val
        elif tag == '隐喻':
            metaphors.append(val)
        elif tag == '抑郁':
            depression_flag = 1
    s = f"[症状:S2={symptoms['S2']},S3={symptoms['S3']},S9={symptoms['S9']}]"
    m = f"[隐喻:{','.join(metaphors)}]" if metaphors else ""
    d = "[抑郁]" if depression_flag else ""
    return s + m + d


def generate_structured_response(expert_input, original_text):
    prompt = (
        "请根据以下信息仅返回案例分析和回复建议，格式为：\n"
        "案例分析：...\n回复建议：...\n\n"
        f"输入内容：{expert_input} {original_text}\n"
        "输出：\n"
    )
    inputs = gen_tokenizer(prompt, return_tensors="pt").to(gen_model.device)
    output = gen_model.generate(
        **inputs, max_new_tokens=128, temperature=0.2,
        top_p=0.85, repetition_penalty=1.2, num_beams=3,
        eos_token_id=gen_tokenizer.eos_token_id
    )
    text = gen_tokenizer.decode(output[0], skip_special_tokens=True)
    return text.split("输出：",1)[-1].strip()


def parse_ai_response(resp):
    a = re.search(r'案例分析[:：]\s*(.*?)(?=(回复建议[:：]|$))', resp, re.DOTALL)
    s = re.search(r'回复建议[:：]\s*(.*)', resp, re.DOTALL)
    ana = a.group(1).strip() if a else ''
    sug = s.group(1).strip().split('。')[0] + '。' if s else ''
    return ana, sug

# ================ Flask Web GUI ================
from flask import Flask, request, render_template_string
app = Flask(__name__)

HTML = '''
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>心理案例分析</title>
  <style>
    body { font-family: "Helvetica Neue", Arial, sans-serif; background: #f0f2f5; margin: 0; padding: 0; }
    .container { max-width: 600px; margin: 50px auto; background: #fff; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); padding: 20px; }
    h1 { text-align: center; color: #333; margin-bottom: 20px; }
    textarea { width: 100%; padding: 10px; border: 1px solid #ccc; border-radius: 4px; resize: vertical; font-size: 14px; }
    button { background: #007bff; color: #fff; border: none; padding: 10px 20px; border-radius: 4px; font-size: 16px; cursor: pointer; }
    button:hover { background: #0056b3; }
    .result { margin-top: 20px; }
    .card { background: #fafafa; border-left: 4px solid #007bff; padding: 15px; margin-bottom: 15px; border-radius: 4px; }
    .card h2 { margin: 0 0 10px; color: #007bff; font-size: 18px; }
    .card p { margin: 0; color: #555; line-height: 1.6; }
  </style>
</head>
<body>
  <div class="container">
    <h1>基于BERT-Qwen的抑郁心理疏导系统</h1>
    <form method="post">
      <textarea name="text" rows="4" placeholder="请输入文本..."></textarea><br><br>
      <div style="text-align:center;"><button type="submit">分析</button></div>
    </form>
    {% if analysis %}
      <div class="result">
        <div class="card">
          <h2>=== 案例分析 ===</h2>
          <p>案例分析：{{ analysis }}</p>
        </div>
        <div class="card">
          <h2>=== 回复建议 ===</h2>
          <p>回复建议：{{ suggestion }}</p>
        </div>
      </div>
    {% endif %}
  </div>
</body>
</html>
'''

@app.route('/', methods=['GET','POST'])
def index():
    analysis = suggestion = None
    if request.method=='POST':
        text = request.form['text']
        labels = classify_text(text, cls_model, cls_tokenizer)
        expert = construct_expert_input(labels)
        resp = generate_structured_response(expert, text)
        analysis, suggestion = parse_ai_response(resp)
    return render_template_string(HTML, analysis=analysis, suggestion=suggestion)

if __name__ == '__main__':
    # 关闭自动重载，避免子进程导入错误
    app.run(host='127.0.0.1', port=5000, debug=True, use_reloader=False)