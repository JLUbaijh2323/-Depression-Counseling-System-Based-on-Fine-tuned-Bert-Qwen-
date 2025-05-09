# -*- coding: utf-8 -*-
import re
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline
)
from peft import PeftModel

# ================= 配置参数 =================
ID2TAG = {
    'LABEL_0': ('S2', 1),   # 情绪低落
    'LABEL_1': ('S3', 1),   # 睡眠障碍
    'LABEL_2': ('S9', 1),   # 自杀倾向
    'LABEL_3': ('隐喻', '自然现象隐喻'),
    'LABEL_4': ('隐喻', '机械故障隐喻'),
    'LABEL_5': ('隐喻', '空间压迫隐喻'),
    'LABEL_6': ('抑郁', 1)  # 独立标签
}

MODEL_PATHS = {
    "distilbert": "main/distilbert_final_model",
    "qwen_base": r"C:\Users\baijh2323\项目\项目\认知科学导论期末项目\main\QwenQwen2.5-0.5B-Instruct",
    "lora": r"C:\Users\baijh2323\项目\项目\认知科学导论期末项目\main\Qwen2.5-0.5B-Instruct\lora\train_2025-05-06-18-10-37\checkpoint-180"
}

# ================= 核心函数 =================
def classify_text(text, model, tokenizer, threshold=0.5):
    """
    使用多标签分类模型对文本进行分类，返回置信度大于 threshold 的标签列表
    """
    classifier = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1,
        top_k=None
    )
    results = classifier(text)[0]
    labels = [item['label'] for item in results if item['score'] > threshold]
    return labels


def construct_expert_input(labels):
    """构建与微调数据完全一致的结构化输入"""
    symptoms = {'S2': 0, 'S3': 0, 'S9': 0}
    metaphors = []
    depression_flag = 0
    
    for label in labels:
        tag_type, value = ID2TAG.get(label, (None, None))
        if tag_type in symptoms:
            symptoms[tag_type] = value
        elif tag_type == '隐喻':
            metaphors.append(value)
        elif tag_type == '抑郁':
            depression_flag = 1

    symptom_str = f"[症状:S2={symptoms['S2']},S3={symptoms['S3']},S9={symptoms['S9']}]"
    metaphor_str = f"[隐喻:{','.join(metaphors)}]" if metaphors else ""
    depression_str = "[抑郁]" if depression_flag else ""
    return f"{symptom_str}{metaphor_str}{depression_str}"


def generate_structured_response(model, tokenizer, expert_input, original_text):
    """生成严格结构化的响应"""
    prompt = (
        f"请根据以下信息仅返回案例分析和回复建议，格式为：\n"
        f"案例分析：...\n回复建议：...\n\n"
        f"输入内容：{expert_input} {original_text}\n"
        f"输出：\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=128,
        temperature=0.2,
        top_p=0.85,
        repetition_penalty=1.2,
        num_beams=3,
        eos_token_id=tokenizer.eos_token_id
    )
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated_text.split("输出：", 1)[-1].strip()


def parse_ai_response(response):
    """解析模型输出中的案例分析和回复建议，且只取首句建议"""
    analysis = re.search(r'案例分析[:：]\s*(.*?)(?=(回复建议[:：]|$))', response, re.DOTALL)
    suggestion = re.search(r'回复建议[:：]\s*(.*)', response, re.DOTALL)
    analysis_text = analysis.group(1).strip() if analysis else None
    suggestion_text = suggestion.group(1).strip() if suggestion else None
    # 只保留建议的第一句话
    if suggestion_text:
        first_sentence = suggestion_text.split('。')[0]
        suggestion_text = first_sentence + '。'
    return analysis_text, suggestion_text


def main():
    text = input("请输入文本: ").strip()
    if not text:
        print("输入不能为空")
        return

    print("加载分类模型...")
    distilbert = AutoModelForSequenceClassification.from_pretrained(MODEL_PATHS["distilbert"])
    tokenizer_cls = AutoTokenizer.from_pretrained(MODEL_PATHS["distilbert"])

    labels = classify_text(text, distilbert, tokenizer_cls)
    print("检测到标签:", labels)

    expert_input = construct_expert_input(labels)
    print("\n结构化输入:", expert_input)

    print("\n加载Qwen模型...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATHS["qwen_base"],
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    qwen_model = PeftModel.from_pretrained(base_model, MODEL_PATHS["lora"])
    tokenizer_gen = AutoTokenizer.from_pretrained(MODEL_PATHS["qwen_base"])

    print("\n生成结构化回复...")
    response = generate_structured_response(qwen_model, tokenizer_gen, expert_input, text)

    analysis, suggestion = parse_ai_response(response)
    if analysis and suggestion:
        print("\n\033[1;34m=== 案例分析 ===\033[0m")
        print(f"案例分析：{analysis}")
        print("\n\033[1;32m=== 回复建议 ===\033[0m")
        print(f"回复建议：{suggestion}")
    else:
        print("\n\033[1;31m生成格式异常，原始响应:\033[0m\n", response)

if __name__ == "__main__":
    main()