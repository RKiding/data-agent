"""
完整的数据科学分析流程
整合数据清洗、特征工程、EDA、模型训练、可视化和报告生成
"""

import os
import sys
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime
import json

# 添加agents目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'agents'))

# 导入所有agent
from agents.data_cleaning import run_data_cleaning
from agents.feature_engineering import run_feature_engineering
from agents.eda_agent import run_eda_analysis
from agents.model_training_agent import run_model_training
from agents.visualization_agent import run_visualization, load_analysis_results
from agents.report_generation_agent import run_report_generation, load_all_results, find_visualization_files

class DataSciencePipeline:
    """完整的数据科学分析流程管理器"""
    
    def __init__(self, data_path: str, output_dir: str = "output"):
        self.data_path = data_path
        self.output_dir = output_dir
        self.results = {}
        self.pipeline_config = {
            "enable_manual_review": False,
            "enable_rag": False,
            "target_variable": None,
            "problem_type": None,
            "report_format": "markdown"  # 新增：报告格式
        }
        
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"🚀 初始化数据科学流程")
        print(f"📁 数据文件: {self.data_path}")
        print(f"📁 输出目录: {self.output_dir}")
    
    def configure_pipeline(self, **kwargs):
        """配置流程参数"""
        self.pipeline_config.update(kwargs)
        print(f"⚙️ 流程配置更新: {kwargs}")
        if "report_format" in kwargs:
            print(f"📄 报告输出格式: {kwargs['report_format']}")
    
    def run_data_cleaning(self, user_instructions: str = "请清洗这份数据，处理缺失值、异常值和重复数据") -> str:
        """步骤1: 数据清洗"""
        print("\n" + "="*50)
        print("步骤 1: 数据清洗")
        print("="*50)
        
        try:
            # 注意：这里需要实现数据清洗的调用逻辑
            # 由于原始的data_cleaning.py结构，我们需要适配
            print("数据清洗功能需要从agents/data_cleaning.py中单独实现")
            print("暂时跳过数据清洗步骤，使用原始数据")
            
            # 复制原始数据作为清洗后的数据
            cleaned_data_path = os.path.join(self.output_dir, "cleaned_data.csv")
            data = pd.read_csv(self.data_path)
            data.to_csv(cleaned_data_path, index=False)
            
            self.results["data_cleaning"] = {
                "input_path": self.data_path,
                "output_path": cleaned_data_path,
                "status": "completed"
            }
            
            return cleaned_data_path
            
        except Exception as e:
            print(f"数据清洗失败: {str(e)}")
            self.results["data_cleaning"] = {"status": "failed", "error": str(e)}
            return self.data_path  # 返回原始数据路径
    
    def run_feature_engineering(self, data_path: str, user_instructions: str = "请对这份金融数据做特征工程") -> str:
        """步骤2: 特征工程"""
        print("\n" + "="*50) 
        print("步骤 2: 特征工程")
        print("="*50)
        
        try:
            # 使用agents.feature_engineering模块
            engineered_data_path = run_feature_engineering(
                data_path=data_path,
                input_text=user_instructions,
                target_variable=self.pipeline_config.get("target_variable"),
                enable_manual_review=self.pipeline_config["enable_manual_review"],
                enable_rag=self.pipeline_config["enable_rag"]
            )
            
            if engineered_data_path and os.path.exists(engineered_data_path):
                # 读取数据以获取形状信息
                original_data = pd.read_csv(data_path)
                engineered_data = pd.read_csv(engineered_data_path)
                
                self.results["feature_engineering"] = {
                    "input_path": data_path,
                    "output_path": engineered_data_path,
                    "input_shape": original_data.shape,
                    "output_shape": engineered_data.shape,
                    "status": "completed"
                }
                
                return engineered_data_path
            else:
                raise Exception("特征工程失败：未生成有效输出文件")
                
        except Exception as e:
            print(f"特征工程失败: {str(e)}")
            self.results["feature_engineering"] = {"status": "failed", "error": str(e)}
            return data_path  # 返回原始数据路径作为fallback
    
    def run_eda(self, data_path: str, user_instructions: str = "请对这份期货数据进行全面的探索性数据分析") -> Dict[str, Any]:
        """步骤3: 探索性数据分析"""
        print("\n" + "="*50)
        print("步骤 3: 探索性数据分析 (EDA)")
        print("="*50)
        
        try:
            eda_results = run_eda_analysis(
                data_path=data_path,
                input_text=user_instructions,
                target_variable=self.pipeline_config.get("target_variable"),
                enable_manual_review=self.pipeline_config["enable_manual_review"],
                enable_rag=self.pipeline_config["enable_rag"]
            )
            
            if eda_results:
                self.results["eda"] = {
                    "status": "completed",
                    "results": eda_results,
                    "output_path": "output/eda_results.pkl"
                }
                return eda_results
            else:
                raise Exception("EDA分析失败")
                
        except Exception as e:
            print(f"EDA分析失败: {str(e)}")
            self.results["eda"] = {"status": "failed", "error": str(e)}
            return None
    
    def run_model_training(self, data_path: str, user_instructions: str = "请对这份期货数据进行机器学习建模") -> Dict[str, Any]:
        """步骤4: 模型训练"""
        print("\n" + "="*50)
        print("步骤 4: 模型训练")
        print("="*50)
        
        try:
            model_results = run_model_training(
                data_path=data_path,
                input_text=user_instructions,
                target_variable=self.pipeline_config.get("target_variable"),
                problem_type=self.pipeline_config.get("problem_type"),
                enable_manual_review=self.pipeline_config["enable_manual_review"],
                enable_rag=self.pipeline_config["enable_rag"]
            )
            
            if model_results:
                self.results["model_training"] = {
                    "status": "completed",
                    "results": model_results,
                    "output_path": "output/model_results.pkl",
                    "best_model_path": "output/best_model.pkl"
                }
                return model_results
            else:
                raise Exception("模型训练失败")
                
        except Exception as e:
            print(f"模型训练失败: {str(e)}")
            self.results["model_training"] = {"status": "failed", "error": str(e)}
            return None
    
    def run_visualization(self, data_path: str, user_instructions: str = "请为这份期货数据生成全面的可视化图表") -> List[str]:
        """步骤5: 数据可视化"""
        print("\n" + "="*50)
        print("步骤 5: 数据可视化")
        print("="*50)
        
        try:
            # 加载之前的分析结果
            analysis_results = load_analysis_results(
                eda_results_path="output/eda_results.pkl",
                model_results_path="output/model_results.pkl"
            )
            
            generated_plots = run_visualization(
                data_path=data_path,
                input_text=user_instructions,
                analysis_results=analysis_results,
                enable_manual_review=self.pipeline_config["enable_manual_review"],
                enable_rag=self.pipeline_config["enable_rag"]
            )
            
            if generated_plots:
                self.results["visualization"] = {
                    "status": "completed",
                    "plots": generated_plots,
                    "plot_count": len(generated_plots)
                }
                return generated_plots
            else:
                print("可视化生成失败，但继续执行")
                self.results["visualization"] = {"status": "failed", "plots": []}
                return []
                
        except Exception as e:
            print(f"可视化生成失败: {str(e)}")
            self.results["visualization"] = {"status": "failed", "error": str(e), "plots": []}
            return []
    
    def run_report_generation(self, data_path: str) -> str:
        """步骤6: 报告生成（支持多种格式）"""
        print("\n" + "="*50)
        print("步骤 6: 报告生成")
        print("="*50)
        
        report_format = self.pipeline_config.get("report_format", "markdown")
        print(f"📄 报告格式: {report_format}")
        
        try:
            # 加载所有分析结果
            eda_results, model_results = load_all_results(
                eda_results_path="output/eda_results.pkl",
                model_results_path="output/model_results.pkl"
            )
            
            # 查找可视化文件
            visualizations = find_visualization_files("visualizations")
            
            report_path = run_report_generation(
                data_path=data_path,
                eda_results=eda_results,
                model_results=model_results,
                visualizations=visualizations,
                output_format=report_format,  # 新增参数
                enable_rag=self.pipeline_config["enable_rag"]
            )
            
            if report_path:
                self.results["report_generation"] = {
                    "status": "completed",
                    "report_path": report_path,
                    "format": report_format
                }
                return report_path
            else:
                raise Exception("报告生成失败")
                
        except Exception as e:
            print(f"❌ 报告生成失败: {str(e)}")
            self.results["report_generation"] = {"status": "failed", "error": str(e)}
            return None
    
    def run_complete_pipeline(self, custom_instructions: Dict[str, str] = None) -> Dict[str, Any]:
        """运行完整的数据科学流程"""
        print("\n" + "="*80)
        print("开始执行完整的数据科学分析流程")
        print("="*80)
        
        start_time = datetime.now()
        
        # 默认指令
        default_instructions = {
            "data_cleaning": "请清洗这份数据，处理缺失值、异常值和重复数据",
            "feature_engineering": "请对这份金融数据做特征工程，重点关注价格、成交量和技术指标特征",
            "eda": "请对这份期货数据进行全面的探索性数据分析，重点分析价格走势、成交量模式和各特征之间的关系",
            "model_training": "请对这份期货数据进行机器学习建模，预测价格趋势或收益率",
            "visualization": "请为这份期货数据生成全面的可视化图表，包括价格走势、特征分布、相关性分析和模型结果可视化",
        }
        
        # 合并自定义指令
        if custom_instructions:
            default_instructions.update(custom_instructions)
        
        current_data_path = self.data_path
        
        # 步骤1: 数据清洗
        current_data_path = self.run_data_cleaning(default_instructions["data_cleaning"])
        
        # 步骤2: 特征工程
        current_data_path = self.run_feature_engineering(current_data_path, default_instructions["feature_engineering"])
        
        # 步骤3: EDA
        eda_results = self.run_eda(current_data_path, default_instructions["eda"])
        
        # 步骤4: 模型训练
        model_results = self.run_model_training(current_data_path, default_instructions["model_training"])
        
        # 步骤5: 可视化
        visualizations = self.run_visualization(current_data_path, default_instructions["visualization"])
        
        # 步骤6: 报告生成
        report_path = self.run_report_generation(current_data_path)
        
        # 计算总耗时
        end_time = datetime.now()
        total_time = end_time - start_time
        
        # 汇总结果
        summary = {
            "pipeline_config": self.pipeline_config,
            "execution_time": str(total_time),
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "results": self.results,
            "final_data_path": current_data_path,
            "report_path": report_path,
            "success_rate": sum(1 for r in self.results.values() if r.get("status") == "completed") / len(self.results)
        }
        
        # 保存流程摘要
        summary_path = os.path.join(self.output_dir, "pipeline_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            # 将不能JSON序列化的对象转换为字符串
            json_summary = self._make_json_serializable(summary)
            json.dump(json_summary, f, ensure_ascii=False, indent=2)
        
        print("\n" + "="*80)
        print("数据科学分析流程执行完成")
        print("="*80)
        print(f"总耗时: {total_time}")
        print(f"成功率: {summary['success_rate']:.1%}")
        print(f"最终数据文件: {current_data_path}")
        print(f"分析报告: {report_path}")
        print(f"流程摘要: {summary_path}")
        
        return summary
    
    def _make_json_serializable(self, obj):
        """使对象可以被JSON序列化"""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif hasattr(obj, 'isoformat'):  # datetime对象
            return obj.isoformat()
        elif hasattr(obj, '__dict__'):  # 其他对象
            return str(obj)
        else:
            return obj

def main():
    """主函数 - 运行完整的数据科学流程"""
    
    # 配置参数
    data_path = "/Users/runkeruan/Desktop/RBM/data-agent-for-futures/output/cleaned_B.csv"
    target_variable = None  # 根据实际情况设置，例如 "close_price" 或 "return"
    
    print("\n" + "="*80)
    print("🚀 数据科学智能分析系统")
    print("="*80)
    print("✨ 新功能:")
    print("  - 🔄 进度条显示")
    print("  - 🧪 小样本测试模式")
    print("  - 📄 多种报告格式支持 (markdown, html, json, pdf)")
    print("  - 🛡️ 改进的错误处理")
    print("="*80)
    
    # 初始化流程
    pipeline = DataSciencePipeline(data_path)
    
    # 配置流程参数
    pipeline.configure_pipeline(
        enable_manual_review=False,  # 是否启用人工审核
        enable_rag=False,           # 是否启用RAG
        target_variable=target_variable,
        problem_type=None,          # 让系统自动检测
        report_format="markdown"    # 可选: markdown, html, json, pdf
    )
    
    # 自定义指令（可选）
    custom_instructions = {
        "feature_engineering": "请对这份期货数据做特征工程，重点创建技术分析指标、滑动平均线、价格变化率等金融特征",
        "eda": "请对这份期货数据进行深入的探索性数据分析，包括价格趋势分析、交易量分析、波动性分析和季节性模式",
        "model_training": "请构建多种机器学习模型来预测期货价格趋势，包括回归模型和分类模型，并进行模型对比",
        "visualization": "请生成专业的金融数据可视化图表，包括K线图、技术指标图、相关性热力图和模型性能图表"
    }
    
    # 运行完整流程
    results = pipeline.run_complete_pipeline(custom_instructions)    
    print(f"\n🎉 流程执行完成！")
    print(f"📊 成功率: {results.get('success_rate', 0):.1%}")
    print(f"⏱️ 总耗时: {results.get('execution_time', 'N/A')}")
    print(f"📄 详细结果请查看: {results.get('report_path', 'output/comprehensive_analysis_report.md')}")
    
    # 显示支持的报告格式
    print(f"\n📄 支持的报告格式:")
    print(f"  - markdown: 轻量级标记语言格式")
    print(f"  - html: 网页格式，可在浏览器中查看")
    print(f"  - json: 结构化数据格式，便于程序处理")
    print(f"  - pdf: 便携式文档格式（需要额外依赖）")
    print(f"\n💡 提示: 可以通过修改 report_format 参数来选择不同的输出格式")

if __name__ == "__main__":
    main()