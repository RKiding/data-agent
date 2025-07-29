from langchain.chat_models import init_chat_model
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 检查API密钥
if not os.getenv('DEEPSEEK_API_KEY'):
    print("⚠️ 警告: 未设置DEEPSEEK_API_KEY环境变量")
    llm = None
else:
    llm = init_chat_model("deepseek:deepseek-chat", temperature=0.7)

def generate_rag_query(context: str, task: str) -> str:
    if llm is None:
        return ""
    
    prompt = f"""你是一个金融数据分析专家。请根据以下任务和上下文，生成一个用于检索知识库的简洁中文查询：
任务: {task}
上下文: {context}
请只输出检索用的查询短语，不要多余解释。"""
    return llm.invoke(prompt).content.strip()

def rag_retrieve_agentic(context: str, task: str) -> str:
    if llm is None:
        return ""
        
    query = generate_rag_query(context, task)
    kb_dir = "knowledge_base"
    if not os.path.exists(kb_dir):
        return ""
    # 全局设置
    Settings.llm = llm
    # TODO：设置向量存储和分片参数
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-zh-v1.5", embed_batch_size=10)
    Settings.chunk_size = 512
    Settings.chunk_overlap = 50
    # TODO：分不同的文件类型进行分片
    docs = SimpleDirectoryReader(input_dir=kb_dir).load_data()
    index = VectorStoreIndex.from_documents(docs)
    query_engine = index.as_query_engine(similarity_top_k=3, response_mode="tree_summarize")
    response = query_engine.query(query)
    return f"【RAG检索Query: {query}】\n{response}"