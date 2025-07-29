import os
import pandas as pd
from typing import Optional, List, TypedDict, Any, Dict
from tqdm import tqdm
import time
import json

from pydantic import BaseModel, Field
from langgraph.graph import Graph
from langchain.chat_models import init_chat_model

from prompt import (
    report_generation_prompt,
)
from utils.dataframe import get_dataframe_summary
from utils.regex import relocate_imports_inside_function
from utils.rag_tool import rag_retrieve_agentic
from utils.llm_utils import get_conservative_llm
from parsers import PythonOutputParser

from dotenv import load_dotenv

load_dotenv()

# 使用统一的LLM初始化（保守型，避免不稳定）
llm = get_conservative_llm()

DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')

class ReportState(TypedDict):
    data: pd.DataFrame
    data_summary: str
    eda_results: Optional[Dict[str, Any]]
    model_results: Optional[Dict[str, Any]]
    visualizations: Optional[List[str]]
    report_code: str
    report_path: Optional[str]
    error: Optional[str]
    retry_count: int
    output_format: str  # 新增：输出格式

def get_data_summary(state: ReportState) -> ReportState:
    data_summary = get_dataframe_summary(state["data"])
    state["data_summary"] = "\n\n".join(data_summary)
    return state

def generate_report_code(state: ReportState) -> ReportState:
    if not llm:
        state["error"] = "LLM未正确初始化"
        # 使用fallback代码
        state["report_code"] = create_fallback_report_code(state.get('output_format', 'markdown'))
        return state
        
    print("\n=== 开始生成报告代码 ===")
    
    # 准备分析结果信息（简化以减少API负载）
    analysis_summary = {
        "has_eda": state.get("eda_results") is not None,
        "has_models": state.get("model_results") is not None,
        "num_visualizations": len(state.get("visualizations", [])),
        "data_shape": state["data"].shape
    }
    
    # 简化的prompt，避免过大的请求
    simplified_prompt = f"""
    You are a Data Science Report Generation Agent. Create a generate_report() function that generates 
    a comprehensive analysis report in {state.get('output_format', 'markdown')} format.
    
    The report should include:
    * Executive Summary
    * Data Overview (shape: {analysis_summary['data_shape']})
    * Analysis Results Summary
    * Key Insights and Recommendations
    
    Available data:
    - EDA Results: {'Yes' if analysis_summary['has_eda'] else 'No'}
    - Model Results: {'Yes' if analysis_summary['has_models'] else 'No'}
    - Visualizations: {analysis_summary['num_visualizations']} files
    
    Return Python code in ```python``` format:

    def generate_report(data, eda_results, model_results, visualizations, output_file="report.md"):
        import pandas as pd
        import numpy as np
        from datetime import datetime
        import os
        
        # Your report generation code here
        # Support formats: markdown (.md), html (.html), json (.json)
        
        return output_file
    """
    
    try:
        with tqdm(total=1, desc="🤖 生成报告代码") as pbar:
            # 添加重试机制
            for attempt in range(3):
                try:
                    time.sleep(1)  # 避免API限制
                    output = llm.invoke(simplified_prompt)
                    break
                except Exception as e:
                    if attempt == 2:
                        raise e
                    print(f"重试 {attempt + 1}/3: {str(e)}")
                    time.sleep(2)
            pbar.update(1)
        
        report_code = PythonOutputParser().parse(output.content)
        
        if "def generate_report" not in report_code:
            print("警告: 生成的代码中未找到 generate_report 函数定义")
            # 提供fallback代码
            report_code = create_fallback_report_code(state.get('output_format', 'markdown'))
        
        state["report_code"] = report_code
        print("\n生成的代码:")
        print(state["report_code"][:500] + "..." if len(state["report_code"]) > 500 else state["report_code"])
        os.makedirs("tmp", exist_ok=True)
        with open("tmp/generate_report.py", "w") as f:
            f.write(state["report_code"])
            
    except Exception as e:
        print(f"API调用失败: {str(e)}")
        state["error"] = f"API调用失败: {str(e)}"
        # 使用fallback代码
        state["report_code"] = create_fallback_report_code(state.get('output_format', 'markdown'))
        
    return state

def create_fallback_report_code(output_format: str = 'markdown') -> str:
    """创建备用的报告生成代码"""
    if output_format.lower() == 'html':
        return '''
def generate_report(data, eda_results, model_results, visualizations, output_file="report.html"):
    import pandas as pd
    from datetime import datetime
    import os
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>数据科学分析报告</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1, h2, h3 {{ color: #333; }}
            .summary {{ background: #f0f0f0; padding: 20px; border-radius: 5px; }}
        </style>
    </head>
    <body>
        <h1>数据科学分析报告</h1>
        <div class="summary">
            <h2>执行摘要</h2>
            <p>数据集包含 {data.shape[0]} 行和 {data.shape[1]} 列。</p>
            <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <h2>数据概览</h2>
        <p>数据形状: {data.shape}</p>
        <p>数据列: {list(data.columns)}</p>
        
        <h2>分析结果</h2>
        <p>EDA分析结果: {'已完成' if eda_results else '未执行'}</p>
        <p>模型训练结果: {'已完成' if model_results else '未执行'}</p>
        <p>可视化图表数量: {len(visualizations) if visualizations else 0}</p>
    </body>
    </html>
    """
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return output_file
        '''
    elif output_format.lower() == 'json':
        return '''
def generate_report(data, eda_results, model_results, visualizations, output_file="report.json"):
    import pandas as pd
    import json
    from datetime import datetime
    import os
    
    report_data = {
        "title": "数据科学分析报告",
        "generated_at": datetime.now().isoformat(),
        "data_summary": {
            "shape": data.shape,
            "columns": list(data.columns),
            "dtypes": data.dtypes.to_dict()
        },
        "analysis_summary": {
            "eda_completed": eda_results is not None,
            "models_trained": model_results is not None,
            "visualizations_count": len(visualizations) if visualizations else 0
        },
        "key_insights": [
            f"数据集包含 {data.shape[0]} 行和 {data.shape[1]} 列",
            f"分析完成度: {'EDA完成' if eda_results else 'EDA未完成'}, {'模型训练完成' if model_results else '模型训练未完成'}"
        ]
    }
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)
    
    return output_file
        '''
    else:  # markdown (default)
        return '''
def generate_report(data, eda_results, model_results, visualizations, output_file="report.md"):
    import pandas as pd
    from datetime import datetime
    import os
    
    content = f"""# 数据科学分析报告

## 执行摘要
- 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- 数据集规模: {data.shape[0]} 行 × {data.shape[1]} 列
- 分析状态: {'✅ EDA已完成' if eda_results else '❌ EDA未完成'}, {'✅ 模型训练已完成' if model_results else '❌ 模型训练未完成'}

## 数据概览
- **数据形状**: {data.shape}
- **数据列**: {', '.join(data.columns)}
- **数据类型**: 
{chr(10).join([f'  - {col}: {dtype}' for col, dtype in data.dtypes.items()])}

## 分析结果摘要
### 探索性数据分析 (EDA)
{'✅ EDA分析已完成，包含统计摘要、相关性分析等。' if eda_results else '❌ EDA分析未执行。'}

### 机器学习建模
{'✅ 模型训练已完成，包含多个模型的性能对比。' if model_results else '❌ 模型训练未执行。'}

### 数据可视化
- 生成图表数量: {len(visualizations) if visualizations else 0}
{chr(10).join([f'- {viz}' for viz in visualizations]) if visualizations else '- 无可视化图表生成'}

## 关键洞察
1. 数据集包含 {data.shape[0]} 条记录和 {data.shape[1]} 个特征
2. {'数据质量良好，适合进行机器学习建模' if data.shape[0] > 100 else '数据集较小，建议收集更多数据'}
3. {'建议关注模型性能最好的算法进行进一步优化' if model_results else '建议完成模型训练以获得预测能力'}

## 建议和后续步骤
1. **数据质量**: 持续监控数据质量，确保数据的完整性和准确性
2. **模型优化**: {'基于当前结果进行超参数调优' if model_results else '完成模型训练和评估'}
3. **业务应用**: 将最佳模型部署到生产环境中进行实际预测

---
*报告由AI数据科学系统自动生成*
"""

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return output_file
        '''

def execute_report_generation(state: ReportState) -> ReportState:
    print("\n=== 开始执行报告生成 ===")
    try:
        local_namespace = {}
        exec(state["report_code"], globals(), local_namespace)
        if 'generate_report' not in local_namespace:
            raise NameError("未能成功定义 generate_report 函数")
        generate_report = local_namespace['generate_report']
        
        # 确定输出文件扩展名
        format_ext = {
            'markdown': '.md',
            'html': '.html', 
            'json': '.json'
        }
        ext = format_ext.get(state.get('output_format', 'markdown'), '.md')
        output_file = f"output/comprehensive_analysis_report{ext}"
        
        # 执行报告生成
        with tqdm(total=1, desc="📝 生成报告") as pbar:
            state["report_path"] = generate_report(
                state["data"], 
                state.get("eda_results"), 
                state.get("model_results"),
                state.get("visualizations", []),
                output_file=output_file
            )
            pbar.update(1)
            
        state["error"] = None
        
        print("✅ 报告生成成功完成")
        print(f"📄 报告已保存到: {state['report_path']}")
            
    except Exception as e:
        state["error"] = str(e)
        print(f"❌ 报告生成过程出错: {state['error']}")
        print("报告代码:")
        print(state["report_code"][:300] + "..." if len(state["report_code"]) > 300 else state["report_code"])
        
    return state

def handle_error(state: ReportState, enable_rag: bool = True) -> ReportState:
    print(f"\n=== 处理错误 (第 {state['retry_count'] + 1} 次重试) ===")
    if state["error"] and state["retry_count"] < 3:
        print(f"当前错误: {state['error']}")
        # 简化错误修复，减少API负载
        if "API调用失败" in state["error"]:
            print("检测到API错误，使用fallback报告生成")
            state["report_code"] = create_fallback_report_code(state.get('output_format', 'markdown'))
        
        state["retry_count"] += 1
        return state
    print("达到最大重试次数或无错误需要处理")
    return state

def complete_report_graph(enable_rag: bool = True) -> Graph:
    workflow = Graph()
    workflow.add_node("get_summary", get_data_summary)
    workflow.add_node("generate_code", generate_report_code)
    workflow.add_node("execute_report", execute_report_generation)
    workflow.add_node("handle_error", lambda state: handle_error(state, enable_rag))
    workflow.add_node("end", lambda x: x)

    workflow.add_edge("get_summary", "generate_code")
    workflow.add_edge("generate_code", "execute_report")
    workflow.add_conditional_edges(
        "execute_report",
        lambda x: x["error"] is not None and x["retry_count"] < 3,
        {
            True: "handle_error",
            False: "end"
        }
    )
    workflow.add_edge("handle_error", "generate_code")
    workflow.set_entry_point("get_summary")
    workflow.set_finish_point("end")
    return workflow.compile()

def load_all_results(eda_results_path: str = "output/eda_results.pkl", 
                    model_results_path: str = "output/model_results.pkl"):
    """加载所有分析结果"""
    eda_results = None
    model_results = None
    
    try:
        import pickle
        if os.path.exists(eda_results_path):
            with open(eda_results_path, "rb") as f:
                eda_results = pickle.load(f)
            print(f"✅ 已加载EDA结果: {eda_results_path}")
        else:
            print(f"⚠️ 未找到EDA结果文件: {eda_results_path}")
    except Exception as e:
        print(f"❌ 加载EDA结果失败: {e}")
    
    try:
        import pickle
        if os.path.exists(model_results_path):
            with open(model_results_path, "rb") as f:
                model_results = pickle.load(f)
            print(f"✅ 已加载模型结果: {model_results_path}")
        else:
            print(f"⚠️ 未找到模型结果文件: {model_results_path}")
    except Exception as e:
        print(f"❌ 加载模型结果失败: {e}")
        
    return eda_results, model_results

def find_visualization_files(visualization_dir: str = "visualizations") -> List[str]:
    """查找可视化文件"""
    visualization_files = []
    if os.path.exists(visualization_dir):
        for file in os.listdir(visualization_dir):
            if file.endswith(('.png', '.jpg', '.jpeg', '.svg', '.pdf')):
                visualization_files.append(os.path.join(visualization_dir, file))
    print(f"找到 {len(visualization_files)} 个可视化文件")
    return visualization_files

def run_report_generation(data_path: str, eda_results: Dict[str, Any] = None, 
                         model_results: Dict[str, Any] = None, 
                         visualizations: List[str] = None, 
                         output_format: str = 'markdown',
                         enable_rag: bool = True) -> str:
    """
    运行报告生成的主函数
    
    Args:
        data_path: 数据文件路径
        eda_results: EDA分析结果
        model_results: 模型训练结果
        visualizations: 可视化文件列表
        output_format: 输出格式 ('markdown', 'html', 'json', 'pdf')
        enable_rag: 是否启用RAG
    
    Returns:
        生成的报告文件路径
    """
    data = pd.read_csv(data_path)
    
    initial_state = ReportState(
        data=data,
        data_summary="",
        eda_results=eda_results,
        model_results=model_results,
        visualizations=visualizations or [],
        report_code="",
        report_path=None,
        error=None,
        retry_count=0,
        output_format=output_format.lower()
    )
    
    graph = complete_report_graph(enable_rag=enable_rag)
    
    print(f"\n=== 报告生成工作流图 (格式: {output_format}) ===")
    graph.get_graph().print_ascii()
    
    final_state = graph.invoke(initial_state)
    
    if final_state["report_path"] is not None:
        print(f"\n=== 报告生成成功完成 ===")
        print(f"📄 报告格式: {output_format}")
        print(f"📄 报告路径: {final_state['report_path']}")
        
        # 如果是PDF格式，尝试转换（需要额外依赖）
        if output_format.lower() == 'pdf' and final_state["report_path"].endswith('.md'):
            try:
                pdf_path = convert_markdown_to_pdf(final_state["report_path"])
                if pdf_path:
                    return pdf_path
            except Exception as e:
                print(f"⚠️ PDF转换失败，返回Markdown版本: {e}")
        
        return final_state["report_path"]
    else:
        print("❌ 报告生成失败")
        return None

def convert_markdown_to_pdf(markdown_path: str) -> str:
    """尝试将Markdown转换为PDF（需要安装额外依赖）"""
    try:
        # 尝试使用weasyprint或其他库
        import weasyprint
        from markdown import markdown
        
        with open(markdown_path, 'r', encoding='utf-8') as f:
            markdown_content = f.read()
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1, h2, h3 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            {markdown(markdown_content)}
        </body>
        </html>
        """
        
        pdf_path = markdown_path.replace('.md', '.pdf')
        weasyprint.HTML(string=html_content).write_pdf(pdf_path)
        print(f"✅ PDF转换成功: {pdf_path}")
        return pdf_path
        
    except ImportError:
        print("⚠️ 未安装PDF转换依赖，建议运行: pip install weasyprint markdown")
        return None
    except Exception as e:
        print(f"⚠️ PDF转换失败: {e}")
        return None

if __name__ == "__main__":
    # 示例使用
    data_path = "/Users/runkeruan/Desktop/RBM/data-agent-for-futures/output/engineered_cleaned_B.csv"
    
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
        output_format='markdown',
        enable_rag=False
    ) 