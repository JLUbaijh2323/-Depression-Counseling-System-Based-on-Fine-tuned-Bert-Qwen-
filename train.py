import os
import logging
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import numpy as np
import matplotlib.pyplot as plt

# 增强日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("training_log.txt", mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

# 数据准备（修复分层抽样问题）
os.makedirs("data", exist_ok=True)
os.makedirs("output", exist_ok=True)

dataset = load_dataset("csv", data_files="train.csv")
split_dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)  # 移除stratify参数
dataset = {"train": split_dataset["train"], "test": split_dataset["test"]}

# 显示数据分布
logger.info(f"\n训练集样本数: {len(dataset['train'])}")
logger.info(f"测试集样本数: {len(dataset['test'])}")

# 分词器初始化
tokenizer = AutoTokenizer.from_pretrained("Geotrend/distilbert-base-zh-cased")

def tokenize_function(examples):
    # 验证标签（保持不变）
    label_cols = ["S2", "S3", "S9", "自然现象隐喻", "机械故障隐喻", "空间压迫隐喻", "是否有抑郁"]
    for col in label_cols:
        invalid = [x for x in examples[col] if x not in {0, 1}]
        if invalid:
            logger.error(f"列 {col} 包含无效值: 前5个错误值 {invalid[:5]}...（共{len(invalid)}个）")
            raise ValueError(f"标签列 {col} 存在非0/1值")

    # 处理文本（保持不变）
    tokenized = tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt"  # 返回PyTorch张量
    )
    
    # 生成多标签并转换为浮点型
    labels = []
    for i in range(len(examples["text"])):
        label_vec = [examples[col][i] for col in label_cols]
        labels.append(label_vec)  # 原始标签是0/1的整数列表
    
    # 关键修改：将标签转换为浮点型张量（解决类型不匹配）
    labels = torch.tensor(labels, dtype=torch.float32)  # 显式转换为浮点型
    
    # 记录标签分布（保持不变）
    if not hasattr(tokenize_function, 'logged'):
        logger.info(f"样本标签示例: {labels[:3].tolist()}")
        logger.info(f"正样本分布: {labels.mean(dim=0).tolist()}")
        tokenize_function.logged = True
    
    # 注意：需要将张量转换为列表，以便与datasets库兼容（后续会自动转换为张量）
    return {**tokenized, "labels": labels.tolist()}  # 返回浮点型列表
tokenized_datasets = {
    "train": dataset["train"].map(tokenize_function, batched=True),
    "test": dataset["test"].map(tokenize_function, batched=True)
}

# 改进的模型（添加类别权重）
class CustomDistilBertForSequenceClassification(AutoModelForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.dropout = torch.nn.Dropout(0.3)  # 调整dropout率
        
        # 自动计算类别权重
        all_labels = np.array(tokenized_datasets["train"]["labels"])
        self.class_weights = torch.tensor(
            [(len(all_labels) / (2 * np.sum(all_labels[:, i]))) for i in range(7)],
            dtype=torch.float32
        )
        logger.info(f"\n类别权重: {self.class_weights.numpy()}")

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        outputs = super().forward(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.dropout(outputs.logits)
        
        if labels is not None:
            loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=self.class_weights.to(logits.device))
            loss = loss_fct(logits, labels.float())
            return {"loss": loss, "logits": logits}
        return outputs

model = CustomDistilBertForSequenceClassification.from_pretrained(
    "Geotrend/distilbert-base-zh-cased",
    num_labels=7,
    problem_type="multi_label_classification"
)

def compute_metrics(p):
    # 转换为PyTorch张量（保持原始形状）
    preds = torch.sigmoid(torch.tensor(p.predictions))  # 形状 [样本数, 7]
    labels = torch.tensor(p.label_ids)                  # 形状 [样本数, 7]
    
    # 调整分类阈值（向量化计算，避免循环）
    thresholds = torch.tensor([0.4, 0.4, 0.4, 0.5, 0.5, 0.5, 0.4])  # 转换为张量
    binary_preds = (preds > thresholds).float()  # 形状 [样本数, 7]，布尔转浮点
    
    # 计算标签级准确率（每个标签位置的预测正确比例）
    accuracy = (binary_preds == labels).float().mean().item()  # 逐元素比较后求平均
    
    # 计算宏平均指标（确保输入是二维数组）
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels.numpy(), binary_preds.numpy(), average="macro", zero_division=0
    )
    
    # 各标签指标
    label_metrics = {}
    for i, col in enumerate(["S2", "S3", "S9", "自然现象隐喻", "机械故障隐喻", "空间压迫隐喻", "是否有抑郁"]):
        p, r, f, _ = precision_recall_fscore_support(
            labels[:, i].numpy(), binary_preds[:, i].numpy(), 
            average="binary", zero_division=0
        )
        label_metrics[f"{col}_precision"] = p
        label_metrics[f"{col}_recall"] = r
    
    # 确保返回字典包含所有必要指标
    return {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        **label_metrics
    }
# 优化的训练参数
training_args = TrainingArguments(
    output_dir="output/distilbert_final_model",
    num_train_epochs=15,  # 增加训练轮次
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,  # 增大评估批次
    learning_rate=2e-5,
    weight_decay=0.01,  # 增强正则化
    warmup_ratio=0.1,   # 添加训练预热
    logging_steps=30,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="precision",  # 以精确率为优化目标
    fp16=True,  # 启用混合精度
)

# 增强的训练器（修复可视化数据收集）
class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics_history = {
            'train_loss': [],
            'eval_loss': [],
            'precision': [],
            'recall': [],
            'f1': []
        }
    
    def on_log(self, logs, **kwargs):
        if "loss" in logs:
            self.metrics_history['train_loss'].append(logs["loss"])
    
    def on_epoch_end(self, args, state, control, **kwargs):
        super().on_epoch_end(args, state, control, **kwargs)
        eval_metrics = self.evaluate()
        
        # 记录指标
        for k in ['eval_loss', 'precision', 'recall', 'f1']:
            self.metrics_history[k].append(eval_metrics[k])
        
        # 详细日志
        logger.info("\n" + "="*50)
        logger.info(f"Epoch {int(state.epoch)} 结果:")
        logger.info(f"训练损失: {self.metrics_history['train_loss'][-1]:.4f}")
        logger.info(f"验证损失: {eval_metrics['eval_loss']:.4f}")
        logger.info(f"准确率: {eval_metrics['accuracy']:.4f}")
        logger.info(f"精确率: {eval_metrics['precision']:.4f}")
        logger.info(f"召回率: {eval_metrics['recall']:.4f}")
        logger.info(f"F1分数: {eval_metrics['f1']:.4f}")
        
        # 输出关键指标
        for col in ['是否有抑郁', 'S2', 'S3']:
            logger.info(f"{col} 精确率: {eval_metrics[f'{col}_precision']:.4f}")

# 训练
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    compute_metrics=compute_metrics,
)

logger.info("🚀 开始训练！")
trainer.train()
logger.info("✅ 训练完成！")

# # 最终评估（添加详细指标输出）
# final_metrics = trainer.evaluate()
# logger.info("\n📊 最终评估结果:")
# logger.info(f"验证损失: {final_metrics['eval_loss']:.4f}")
# # logger.info(f"准确率: {final_metrics['accuracy']:.4f}")
# # logger.info(f"精确率: {final_metrics['precision']:.4f}")
# # logger.info(f"召回率: {final_metrics['recall']:.4f}")
# # logger.info(f"F1分数: {final_metrics['f1']:.4f}")

# logger.info("\n各标签精确率:")
# for col in ["S2", "S3", "S9", "是否有抑郁"]:
#     logger.info(f"{col}: {final_metrics[f'{col}_precision']:.4f}")

# # 修复的可视化（确保正确保存）
# def plot_metrics(history):
#     plt.figure(figsize=(14, 10))
    
#     # 损失曲线
#     plt.subplot(2, 2, 1)
#     plt.plot(history['train_loss'], label='Train Loss')
#     plt.plot(history['eval_loss'], label='Eval Loss')
#     plt.title('Training and Validation Loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.legend()
    
#     # 精确率-召回率曲线
#     plt.subplot(2, 2, 2)
#     plt.plot(history['precision'], label='Precision')
#     plt.plot(history['recall'], label='Recall')
#     plt.title('Precision-Recall Curve')
#     plt.xlabel('Epoch')
#     plt.ylabel('Score')
#     plt.legend()
    
#     # F1曲线
#     plt.subplot(2, 2, 3)
#     plt.plot(history['f1'], label='F1 Score')
#     plt.title('F1 Score Trend')
#     plt.xlabel('Epoch')
#     plt.ylabel('F1 Score')
#     plt.legend()
    
#     plt.tight_layout()
#     plt.savefig("training_metrics.png")
#     plt.close()  # 修复图像保存问题

# plot_metrics(trainer.metrics_history)