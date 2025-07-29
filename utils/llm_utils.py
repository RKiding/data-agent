"""
LLM初始化工具
统一管理所有agent的LLM实例，避免重复初始化和API密钥问题
"""

import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

# 加载环境变量
load_dotenv()

def get_llm(temperature: float = 1.0, model_name: str = "deepseek:deepseek-chat"):
    """
    获取LLM实例
    
    Args:
        temperature: 温度参数
        model_name: 模型名称
    
    Returns:
        LLM实例或None（如果API密钥未设置）
    """
    if not os.getenv('DEEPSEEK_API_KEY'):
        print("⚠️ 警告: 未设置DEEPSEEK_API_KEY环境变量")
        return None
    
    try:
        return init_chat_model(model_name, temperature=temperature)
    except Exception as e:
        print(f"❌ LLM初始化失败: {e}")
        return None

# 预定义的LLM实例
def get_default_llm():
    """获取默认LLM实例（temperature=1.0）"""
    return get_llm(temperature=1.0)

def get_conservative_llm():
    """获取保守型LLM实例（temperature=0.7）"""
    return get_llm(temperature=0.7) 