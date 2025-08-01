import os
import pandas as pd
from typing import Optional, List, TypedDict

from pydantic import BaseModel, Field
from langgraph.graph import Graph
from langchain.chat_models import init_chat_model

from prompt import (
    feature_engineering_recommend_steps_prompt,
    feature_engineering_code_fix_prompt,
    feature_engineering_prompt,
)
from utils.dataframe import get_dataframe_summary
from utils.regex import relocate_imports_inside_function
from utils.rag_tool import rag_retrieve_agentic
from utils.llm_utils import get_default_llm
from parsers import PythonOutputParser

from dotenv import load_dotenv

load_dotenv()

# 使用统一的LLM初始化
llm = get_default_llm()

class Router(BaseModel):
    """ a class for feature engineering python code generation """
    output: Optional[str] = Field(
        description="The output text generated by the model."
    )
    steps: Optional[List[str]] = Field(
        description="A list of steps to be followed for processing the input."
    )

# 检查LLM是否可用
if llm is None:
    print("❌ 警告: LLM未正确初始化，特征工程功能可能无法正常工作")
    llm_router = None
else:
    llm_router = llm.with_structured_output(Router)

# llm = init_chat_model("ollama:llama3.2")

class FeatureEngineeringState(TypedDict):
    data_raw: pd.DataFrame
    data_summary: str
    recommended_steps: str
    engineering_code: str
    engineered_data: Optional[pd.DataFrame]
    error: Optional[str]
    retry_count: int
    target_variable: Optional[str]  # 新增

def get_data_summary(state: FeatureEngineeringState) -> FeatureEngineeringState:
    data_summary = get_dataframe_summary(state["data_raw"])
    state["data_summary"] = "\n\n".join(data_summary)
    return state

def recommend_engineering_steps(state: FeatureEngineeringState, input: str, enable_rag: bool = True) -> FeatureEngineeringState:
    if not llm_router:
        state["error"] = "LLM未正确初始化"
        return state
        
    rag_context = ""
    if enable_rag:
        rag_context = rag_retrieve_agentic(
            context=state["data_summary"],
            task=input
        )
    prompt = feature_engineering_recommend_steps_prompt.format(
        user_instructions=input,
        recommended_steps=None,
        all_datasets_summary=state["data_summary"] + ("\n\n" + rag_context if rag_context else "")
    )
    output = llm_router.invoke(prompt)
    state["recommended_steps"] = "\n\n".join(output.steps) if output.steps else ""
    return state

def generate_engineering_code(state: FeatureEngineeringState) -> FeatureEngineeringState:
    if not llm:
        state["error"] = "LLM未正确初始化"
        return state
        
    print("\n=== 开始生成特征工程代码 ===")
    print("基于以下步骤生成代码:")
    print(state["recommended_steps"])
    
    prompt = feature_engineering_prompt.format(
        function_name="feature_engineer",
        recommended_steps=state["recommended_steps"],
        all_datasets_summary=state["data_summary"],
        target_variable=state.get("target_variable", "")  # 新增
    )
    output = llm.invoke(prompt)
    engineered_code = PythonOutputParser().parse(output.content)
    
    if "def feature_engineer" not in engineered_code:
        print("警告: 生成的代码中未找到 feature_engineer 函数定义")
        state["error"] = "生成的代码中缺少 feature_engineer 函数定义"
    else:
        state["engineering_code"] = engineered_code
        print("\n生成的代码:")
        print(state["engineering_code"])
        os.makedirs("tmp", exist_ok=True)
        with open("tmp/feature_engineer.py", "w") as f:
            f.write(state["engineering_code"])
    return state

def execute_engineering(state: FeatureEngineeringState) -> FeatureEngineeringState:
    print("\n=== 开始执行特征工程 ===")
    try:
        local_namespace = {}
        exec(state["engineering_code"], globals(), local_namespace)
        if 'feature_engineer' not in local_namespace:
            raise NameError("未能成功定义 feature_engineer 函数")
        feature_engineer = local_namespace['feature_engineer']
        state["engineered_data"] = feature_engineer(state["data_raw"])
        state["error"] = None
        print("特征工程成功完成")
        print(f"数据形状: {state['engineered_data'].shape}")
    except Exception as e:
        state["error"] = str(e)
        print(f"特征工程过程出错: {state['error']}")
        print("特征工程代码:")
        print(state["engineering_code"])
    return state

def handle_error(state: FeatureEngineeringState, enable_rag: bool = True) -> FeatureEngineeringState:
    if not llm:
        state["error"] = "LLM未正确初始化"
        return state
        
    print(f"\n=== 处理错误 (第 {state['retry_count'] + 1} 次重试) ===")
    if state["error"] and state["retry_count"] < 3:
        print(f"当前错误: {state['error']}")
        rag_context = ""
        if enable_rag:
            rag_context = rag_retrieve_agentic(
                context=state["engineering_code"] + "\n" + (state["error"] or ""),
                task="修复特征工程代码"
            )
        prompt = feature_engineering_code_fix_prompt.format(
            function_name="feature_engineer",
            code_snippet=state["engineering_code"],
            error=state["error"] + ("\n\n" + rag_context if rag_context else "")
        )
        output = llm.invoke(prompt)
        state["engineering_code"] = relocate_imports_inside_function(
            PythonOutputParser().parse(output.content)
        )
        print("\n修复后的代码:")
        print(state["engineering_code"])
        os.makedirs("tmp", exist_ok=True)
        with open("tmp/feature_engineer.py", "w") as f:
            f.write(state["engineering_code"])
        state["retry_count"] += 1
        return state
    print("达到最大重试次数或无错误需要处理")
    return state

def manual_review(state: FeatureEngineeringState, user_instructions: str, enable_rag: bool = True) -> FeatureEngineeringState:
    if not llm_router:
        state["error"] = "LLM未正确初始化"
        return state
        
    print("\n=== 人工审核推荐的特征工程步骤 ===")
    print("推荐的特征工程步骤：")
    print(state["recommended_steps"])
    while True:
        user_input = input("是否接受推荐的特征工程步骤？(y/n)：").strip().lower()
        if user_input in ("y", "n"):
            break
        print("无效输入，请输入 'y' 或 'n'。")
    if user_input == "n":
        while True:
            feedback = input("请简要说明你对推荐步骤的不满或希望改进的地方：\n")
            rag_context = ""
            if enable_rag:
                rag_context = rag_retrieve_agentic(
                    context=state["data_summary"] + "\n用户反馈: " + feedback,
                    task=user_instructions
                )
            prompt = feature_engineering_recommend_steps_prompt.format(
                user_instructions=feedback,
                recommended_steps=state["recommended_steps"],
                all_datasets_summary=state["data_summary"] + ("\n\n" + rag_context if rag_context else "")
            )
            output = llm_router.invoke(prompt)
            state["recommended_steps"] = "\n\n".join(output.steps) if output.steps else ""
            print("根据反馈生成的新特征工程步骤：")
            print(state["recommended_steps"])
            while True:
                user_input = input("是否接受推荐的特征工程步骤？(y/n)：").strip().lower()
                if user_input in ("y", "n"):
                    break
                print("无效输入，请输入 'y' 或 'n'。")
            if user_input == "y":
                break
    return state

def build_graph(user_instructions: str, enable_manual_review: bool = False, enable_rag: bool = True) -> Graph:
    workflow = Graph()
    workflow.add_node("get_summary", get_data_summary)
    workflow.add_node("recommend_steps", lambda state: recommend_engineering_steps(state, user_instructions, enable_rag))
    if enable_manual_review:
        workflow.add_node("manual_review", lambda state: manual_review(state, user_instructions, enable_rag))
    workflow.add_node("generate_code", generate_engineering_code)
    workflow.add_node("execute_engineering", execute_engineering)
    workflow.add_node("handle_error", lambda state: handle_error(state, enable_rag))
    workflow.add_node("end", lambda x: x)

    workflow.add_edge("get_summary", "recommend_steps")
    if enable_manual_review:
        workflow.add_edge("recommend_steps", "manual_review")
        workflow.add_edge("manual_review", "generate_code")
    else:
        workflow.add_edge("recommend_steps", "generate_code")
    workflow.add_edge("generate_code", "execute_engineering")
    workflow.add_conditional_edges(
        "execute_engineering",
        lambda x: x["error"] is not None and x["retry_count"] < 5,
        {
            True: "handle_error",
            False: "end"
        }
    )
    workflow.add_edge("handle_error", "execute_engineering")
    workflow.set_entry_point("get_summary")
    workflow.set_finish_point("end")
    return workflow.compile()

def run_feature_engineering(data_path: str = None, input_text: str = None, 
                            target_variable: str = None, 
                            enable_manual_review: bool = False, 
                            enable_rag: bool = False):
    """
    运行特征工程的主函数
    
    Args:
        data_path: 数据文件路径
        input_text: 用户输入的特征工程需求
        target_variable: 目标变量名称
        enable_manual_review: 是否启用人工审核
        enable_rag: 是否启用RAG
    
    Returns:
        特征工程后的数据文件路径
    """
    # 默认参数
    if not data_path:
        data_path = "/Users/runkeruan/Desktop/RBM/data-agent-for-futures/output/cleaned_B.csv"
    if not input_text:
        input_text = "请对这份金融数据做特征工程"
    
    initial_state = FeatureEngineeringState(
        data_raw=pd.read_csv(data_path),
        data_summary="",
        recommended_steps="",
        engineering_code="",
        engineered_data=None,
        error=None,
        retry_count=0,
        target_variable=target_variable
    )
    
    graph = build_graph(
        user_instructions=input_text, 
        enable_manual_review=enable_manual_review, 
        enable_rag=enable_rag
    )
    
    print("\n=== 特征工程工作流图 ===")
    graph.get_graph().print_ascii()
    final_state = graph.invoke(initial_state)

    if final_state["engineered_data"] is not None:
        print("Shape before engineering:", final_state["data_raw"].shape)
        print("\nShape after engineering:", final_state["engineered_data"].shape)
        original_filename = os.path.splitext(os.path.basename(data_path))[0]
        os.makedirs("output", exist_ok=True)
        output_filename = os.path.join("output", f"engineered_{original_filename}.csv")
        final_state["engineered_data"].to_csv(output_filename, index=False)
        print(f"Engineered data saved to '{output_filename}'")
        return output_filename
    else:
        print("Feature engineering failed after all retries")
        return None

def main():
    """主函数 - 用于直接运行特征工程"""
    return run_feature_engineering(
        enable_manual_review=True,
        enable_rag=False
    )

if __name__ == "__main__":
    main()
