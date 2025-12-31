# 期末实验：IMDB情感分类

本项目实现了一个针对 IMDB 电影评论的二分类情感分析实验，满足“期末实验内容”文档中的要求。代码会自动下载官方 IMDB 数据集并训练一个 TF-IDF + 逻辑回归模型，提供训练与推理脚本。

## 环境准备
1. 安装依赖
   ```bash
   pip install -r requirements.txt
   ```

2. 代码结构
   - `src/data.py`：下载、解压与切分 IMDB 数据集
   - `src/train.py`：模型训练与评估，输出指标与已保存的模型
   - `src/predict.py`：加载已训练模型，对任意文本进行情感分类

> 数据集与模型默认保存在 `data/` 与 `models/` 目录下，已通过 `.gitignore` 排除。

## 训练
以下命令将自动下载数据集，完成训练/验证/测试划分并保存模型与指标：
```bash
PYTHONPATH=. python src/train.py \
  --data-dir data \
  --models-dir models \
  --val-size 0.2 \
  --max-features 40000 \
  --ngram-max 2 \
  --C 3.0 \
  --max-iter 200
```

常用参数：
- `--train-limit` / `--test-limit`：限制样本量以便快速试跑
- `--metrics-path`：自定义指标 JSON 输出路径
- `--model-filename`：自定义保存的模型文件名

训练完成后，终端会展示各划分的准确率，并在 `artifacts/metrics.json` 中记录详细指标。

## 推理
加载训练好的模型并对文本进行情感分类：
```bash
PYTHONPATH=. python src/predict.py models/tfidf_logreg.joblib "This movie was amazing!" "It was a waste of time."
```
输出示例：
```
[positive] This movie was amazing!
[negative] It was a waste of time.
```

## 实验报告撰写建议
- 记录模型设计（特征选择、模型超参）
- 描述训练过程与验证集调参思路
- 汇报测试集结果与误差分析
- 如有创新（使用无标注数据、外部特征或大模型等），请在报告中重点说明
