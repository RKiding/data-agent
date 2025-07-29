# 期货数据智能分析代理系统

## ✨ 最新功能更新 (v2.0)

### 🚀 性能与用户体验优化
- **🔄 进度条显示**: 所有长时间运行的任务现在都有实时进度显示
- **🧪 小样本测试**: 模型训练前会先用100条数据进行快速验证，避免大数据集上的调试成本
- **⚡ 智能重试**: 改进的错误处理和API调用重试机制

### 📄 多格式报告生成
现在支持多种输出格式的分析报告：
- **Markdown** (.md) - 轻量级标记语言格式，易于编辑和版本控制
- **HTML** (.html) - 网页格式，可在浏览器中查看，支持富文本展示
- **JSON** (.json) - 结构化数据格式，便于程序处理和系统集成
- **PDF** (.pdf) - 便携式文档格式，适合正式报告和打印分享

### 🛡️ 稳定性改进
- **API错误恢复**: 当API调用失败时自动使用备用模板生成报告
- **请求优化**: 减少大请求导致的500错误，提高成功率
- **内存管理**: 更好的资源管理和内存使用优化

### 🚀 快速启动命令
```bash
# 基础分析
python run_analysis.py data.csv

# HTML格式报告
python run_analysis.py data.csv --format html

# 测试模式（快速验证）
python run_analysis.py data.csv --test

# PDF报告（需要额外依赖）
python run_analysis.py data.csv --format pdf
```

---

一个完整的AI驱动的数据科学分析流程，专门为期货金融数据设计，提供从数据清洗到模型预测的端到端自动化分析。

## 🚀 主要功能

### 完整数据科学流程
1. **数据清洗** - 自动处理缺失值、异常值和重复数据
2. **特征工程** - 生成金融技术指标和特征
3. **探索性数据分析 (EDA)** - 深入分析数据模式和趋势
4. **机器学习建模** - 多模型训练和比较
5. **数据可视化** - 专业的金融图表生成
6. **报告生成** - 自动生成分析报告

### 核心特性
- 🤖 **AI驱动**: 使用LLM自动生成和优化分析代码
- 🔄 **自动重试**: 智能错误处理和代码修复
- 📊 **专业可视化**: 针对金融数据的专业图表
- 📝 **自动报告**: 生成完整的分析报告
- 🎯 **领域专化**: 专门针对期货/金融数据优化
- 🔧 **高度可配置**: 支持自定义分析步骤和参数

## 📋 系统要求

- Python 3.8+
- 充足的内存（推荐8GB+）
- DeepSeek API密钥或其他LLM提供商

## 🛠️ 安装说明

1. **克隆项目**
```bash
git clone <repository-url>
cd data-agent-for-futures
```

2. **安装依赖**
```bash
pip install -r requirements.txt
```

3. **PDF生成支持（可选）**
```bash
# 安装PDF生成依赖
pip install weasyprint markdown Pygments

# macOS用户可能需要额外安装
brew install pango
```

4. **配置环境变量**
创建 `.env` 文件：
```bash
DEEPSEEK_API_KEY=your_deepseek_api_key_here
# 或其他LLM提供商的API密钥
```

## 🚀 快速开始

### 命令行使用（推荐）

```bash
# 基础分析（Markdown报告）
python run_analysis.py your_data.csv

# HTML格式报告
python run_analysis.py your_data.csv --format html

# PDF格式报告
python run_analysis.py your_data.csv --format pdf

# 测试模式（快速验证）
python run_analysis.py your_data.csv --test

# 完整交互模式
python run_analysis.py your_data.csv --manual --rag
```

### Python API使用

```python
from comprehensive_data_science_pipeline import DataSciencePipeline

# 初始化流程
pipeline = DataSciencePipeline("your_data.csv")

# 配置参数
pipeline.configure_pipeline(
    enable_manual_review=False,     # 人工审核
    enable_rag=False,              # RAG增强
    target_variable="close_price",  # 目标变量
    report_format="html"           # 报告格式
)

# 运行分析
results = pipeline.run_complete_pipeline()
```

## 📁 项目结构

```
data-agent-for-futures/
├── run_analysis.py                       # 🚀 快速启动脚本
├── comprehensive_data_science_pipeline.py # 完整流程管理器
├── main.py                               # 特征工程主文件
├── prompt.py                             # 所有LLM提示模板
├── parsers.py                            # 代码解析器
├── requirements.txt                      # 依赖项
├── README.md                            # 项目说明
├── 
├── agents/                              # 各功能代理
│   ├── data_cleaning.py                # 数据清洗代理
│   ├── feature_engineering.py          # 特征工程代理
│   ├── eda_agent.py                    # EDA分析代理
│   ├── model_training_agent.py         # 模型训练代理
│   ├── visualization_agent.py          # 可视化代理
│   └── report_generation_agent.py      # 报告生成代理
├── 
├── utils/                              # 工具函数
│   ├── dataframe.py                   # 数据框分析工具
│   ├── rag_tool.py                    # RAG检索工具
│   └── regex.py                       # 正则处理工具
├── 
├── output/                            # 输出目录
│   ├── cleaned_data.csv              # 清洗后数据
│   ├── engineered_data.csv           # 特征工程数据
│   ├── eda_results.pkl               # EDA分析结果
│   ├── model_results.pkl             # 模型训练结果
│   ├── best_model.pkl                # 最佳模型
│   ├── comprehensive_analysis_report.* # 分析报告（多格式）
│   └── pipeline_summary.json         # 流程摘要
├── 
├── visualizations/                   # 可视化输出
│   └── *.png                        # 生成的图表
├── 
└── tmp/                             # 临时文件
    └── *.py                         # 生成的代码文件
```

## 🎯 使用场景

### 期货交易分析
- 价格趋势预测
- 技术指标分析
- 风险评估
- 交易策略回测

### 投资研究
- 市场模式识别
- 相关性分析
- 季节性分析
- 基本面分析

### 风险管理
- 波动性分析
- VaR计算
- 压力测试
- 组合优化

## ⚙️ 配置选项

### 流程配置
```python
pipeline.configure_pipeline(
    enable_manual_review=False,    # 启用人工审核
    enable_rag=False,             # 启用RAG检索
    target_variable="price",      # 目标变量
    problem_type="regression",    # 问题类型
    report_format="html"          # 报告格式
)
```

### 自定义指令
```python
custom_instructions = {
    "feature_engineering": "创建技术分析指标",
    "eda": "重点分析价格波动模式",
    "model_training": "使用集成学习方法",
    "visualization": "生成交易信号图表"
}
```

## 📊 输出说明

### 数据文件
- `cleaned_data.csv`: 清洗后的原始数据
- `engineered_data.csv`: 添加特征后的数据

### 分析结果
- `eda_results.pkl`: EDA分析的所有结果
- `model_results.pkl`: 模型性能和预测结果
- `best_model.pkl`: 性能最佳的训练模型

### 报告文件
- `comprehensive_analysis_report.md`: Markdown格式报告
- `comprehensive_analysis_report.html`: HTML格式报告
- `comprehensive_analysis_report.json`: JSON格式报告
- `comprehensive_analysis_report.pdf`: PDF格式报告（可选）

### 可视化
- 价格走势图和技术指标
- 相关性热力图
- 模型性能对比图
- 特征重要性分析图

## 🔧 故障排除

### 常见问题

1. **API错误 (500 Internal Server Error)**
   - ✅ 系统会自动使用备用模板生成报告
   - 建议检查API密钥配置和网络连接

2. **模型训练失败**
   - ✅ 小样本测试会提前发现代码问题
   - 检查数据格式和目标变量设置

3. **PDF生成失败**
   - 安装命令：`pip install weasyprint markdown`
   - ✅ 系统会自动降级到markdown格式

### 调试模式
```bash
# 启用测试模式
python run_analysis.py data.csv --test --skip-check
```

## 💡 最佳实践

### 1. 数据准备
- 确保数据为CSV格式
- 包含时间戳列（用于时间序列分析）
- 数据质量检查（无过多缺失值）

### 2. 性能优化
- 大型数据集建议先使用 `--test` 模式验证
- 复杂分析可以分步骤执行单个agent
- 考虑使用 `--skip-check` 跳过重复的系统检查

### 3. 报告格式选择
- **开发调试**: 使用 `markdown` 格式
- **演示展示**: 使用 `html` 格式
- **系统集成**: 使用 `json` 格式
- **正式报告**: 使用 `pdf` 格式

## 🎉 更新日志

### v2.0 - 2024年最新版本
- ✨ 新增多格式报告生成
- ✨ 新增进度条显示
- ✨ 新增小样本测试功能
- 🛡️ 改进错误处理和重试机制
- 🚀 新增快速启动脚本
- 📚 完善文档和使用指南

### v1.0 - 初始版本
- 🎯 完整数据科学流程
- 🤖 AI驱动的代码生成
- 📊 专业金融数据可视化

## 📞 支持与贡献

如果您遇到问题或有改进建议，欢迎提交Issue或Pull Request。

---

**Happy Analyzing! 🚀📊**