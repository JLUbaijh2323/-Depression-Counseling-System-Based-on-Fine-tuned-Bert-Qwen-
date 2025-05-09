"""
GLM-4-FLASH数据标注完整实现
环境准备：Python 3.8+，需要安装以下库：
pip install pandas requests chardet tqdm
"""

# ========== 第一部分：配置参数 ==========
import pandas as pd
import requests
import json
import time
import os
from tqdm import tqdm
import chardet

# GLM-4-FLASH API配置
API_KEY = "50e9a0f4ae3e41c1b6c8fe6525a8c370.Bvb3xsMmLIOojwIZ"  # 需替换为实际API密钥
API_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"  # 确认最新API地址

# 文件路径配置（注意检查路径中的中文是否需要转码）
TRAIN_FILE = r"数据集构建\数据集构建\datasets\train.csv"  # 已标注数据
PREDICT_FILE = r"数据集构建\数据集构建\datasets\原始数据集.csv"  # 待标注数据
OUTPUT_FILE = r"datasets\result.csv"  # 输出文件

# 模型参数
MAX_TOKENS = 4096  # 根据实际需要调整
TEMPERATURE = 0.3  # 控制生成稳定性（0-1，值越小越稳定）

# ========== 第二部分：数据预处理 ==========
def load_data(file_path):
    """
    加载CSV数据文件
    参数说明：
    file_path -- 文件路径（注意中文路径可能需要转码处理）
    返回值：pandas DataFrame对象
    """
    try:
        # 自动检测文件编码
        with open(file_path, 'rb') as f:
            result = chardet.detect(f.read())
        return pd.read_csv(file_path, encoding=result['encoding'])
    except Exception as e:
        print(f"文件加载失败：{str(e)}")
        exit()

# 加载训练数据（用于few-shot学习）
train_df = load_data(TRAIN_FILE)

# ========== 第三部分：构建提示模板 ==========
def build_prompt(text):
    """
    构建带few-shot示例的提示模板
    参数说明：
    text -- 需要标注的文本
    返回值：完整提示字符串
    """
    # 从训练数据中随机选择3个示例（可根据效果调整数量）
    examples = train_df.sample(300).to_dict('records')
    
    prompt = """你是一个专业的认知科学标注助手。请根据以下示例的格式，对用户描述的心理学状态进行标注。
    
示例格式：
输入文本：[文本内容]
标注结果：
S2：情绪低落 [0/1]
S3：睡眠障碍 [0/1]
S9：自杀倾向 [0/1]
?然现象隐喻 [0/1]
机械故障隐喻 [0/1]
空间压迫隐喻 [0/1]
是否有抑郁 [0/1]

现在请处理以下案例："""
    
    # 添加示例
    for exp in examples:
        prompt += f"\n\n输入文本：{exp['text']}\n标注结果：\n"
        prompt += f"S2：情绪低落 {exp['S2：情绪低落']}\n"
        prompt += f"S3：睡眠障碍 {exp['S3：睡眠障碍']}\n"
        prompt += f"S9：自杀倾向 {exp['S9：自杀倾向']}\n"
        prompt += f"?然现象隐喻 {exp['?然现象隐喻']}\n"
        prompt += f"机械故障隐喻 {exp['机械故障隐喻']}\n"
        prompt += f"空间压迫隐喻 {exp['空间压迫隐喻']}\n"
        prompt += f"是否有抑郁 {exp['是否有抑郁']}"
    
    # 添加当前需要标注的文本
    prompt += f"\n\n输入文本：{text}\n标注结果："
    return prompt

# ========== 第四部分：API调用函数 ==========
def glm4_api_call(prompt):
    """
    GLM-4-FLASH API调用函数
    参数说明：
    prompt -- 构建好的提示文本
    返回值：模型生成的文本
    """
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "glm-4-flash",  # 确认模型名称
        "messages": [{"role": "user", "content": prompt}],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content']
    except Exception as e:
        print(f"API调用失败：{str(e)}")
        return None

# ========== 第五部分：结果解析 ==========
def parse_output(output):
    """
    解析模型输出结果
    参数说明：
    output -- 模型生成的原始文本
    返回值：字典形式的标注结果
    """
    result = {}
    lines = output.split('\n')
    for line in lines:
        if "：" in line:
            key, value = line.split("：", 1)
            key = key.strip()
            value = value.strip()
            # 提取数字标签
            if '0' in value:
                result[key] = 0
            elif '1' in value:
                result[key] = 1
    return result

# ========== 第六部分：主流程 ==========
if __name__ == "__main__":
    # 加载待标注数据
    predict_df = load_data(PREDICT_FILE)
    
    # 创建进度条
    pbar = tqdm(total=len(predict_df), desc="数据标注进度")
    
    # 遍历数据并标注
    results = []
    for idx, row in predict_df.iterrows():
        # 构建提示
        prompt = build_prompt(row['text'])
        
        # 调用API（包含重试机制）
        max_retries = 3
        for attempt in range(max_retries):
            output = glm4_api_call(prompt)
            if output:
                parsed = parse_output(output)
                results.append(parsed)
                break
            else:
                time.sleep(2**attempt)  # 指数退避
        else:
            print(f"第{idx}条数据标注失败")
            results.append({})  # 添加空结果
            
        pbar.update(1)
        time.sleep(1)  # 控制请求频率
    
    pbar.close()
    
    # 合并结果并保存
    result_df = pd.DataFrame(results)
    final_df = pd.concat([predict_df, result_df], axis=1)
    final_df.to_csv(OUTPUT_FILE, index=False, encoding='utf_8_sig')
    print("标注完成！结果已保存至：", OUTPUT_FILE)