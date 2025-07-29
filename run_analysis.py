#!/usr/bin/env python3
"""
数据科学分析系统快速启动脚本
支持多种输出格式、进度条显示、小样本测试等功能
"""

import os
import sys
import argparse
from datetime import datetime
from dotenv import load_dotenv

def check_requirements():
    """检查系统要求"""
    print("🔍 检查系统要求...")
    
    # 检查Python版本
    if sys.version_info < (3, 8):
        print("❌ 需要Python 3.8或更高版本")
        return False
    
    # 检查必要的包 - 使用更准确的包名
    required_packages = [
        ('pandas', 'pandas'), 
        ('numpy', 'numpy'), 
        ('sklearn', 'scikit-learn'), 
        ('matplotlib', 'matplotlib'), 
        ('seaborn', 'seaborn'), 
        ('langchain', 'langchain'), 
        ('langgraph', 'langgraph'), 
        ('tqdm', 'tqdm')
    ]
    
    missing_packages = []
    for import_name, package_name in required_packages:
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package_name)
    
    if missing_packages:
        print(f"❌ 缺少以下包: {', '.join(missing_packages)}")
        print("请运行: pip install -r requirements.txt")
        return False
    
    # 检查API密钥 - 改进检查逻辑
    load_dotenv()  # 确保加载.env文件
    
    if not os.getenv('DEEPSEEK_API_KEY'):
        print("⚠️ 未设置DEEPSEEK_API_KEY环境变量")
        print("请在.env文件中设置API密钥，或使用 --skip-check 跳过此检查")
        return False
    
    print("✅ 系统要求检查通过")
    return True

def setup_directories():
    """设置目录结构"""
    print("📁 设置目录结构...")
    
    directories = [
        "output", "tmp", "visualizations", "agents"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("✅ 目录结构设置完成")

def run_analysis(data_path: str, target_variable: str = None, 
                 report_format: str = "markdown", 
                 enable_manual_review: bool = False,
                 enable_rag: bool = False,
                 test_mode: bool = False):
    """运行数据分析"""
    print(f"\n🚀 开始数据科学分析")
    print(f"📁 数据文件: {data_path}")
    print(f"🎯 目标变量: {target_variable or '自动检测'}")
    print(f"📄 报告格式: {report_format}")
    print(f"🧪 测试模式: {'启用' if test_mode else '禁用'}")
    print(f"👥 人工审核: {'启用' if enable_manual_review else '禁用'}")
    print(f"🔍 RAG功能: {'启用' if enable_rag else '禁用'}")
    
    # 导入主要模块
    from comprehensive_data_science_pipeline import DataSciencePipeline
    
    # 初始化流程
    pipeline = DataSciencePipeline(data_path)
    
    # 配置参数
    config = {
        "enable_manual_review": enable_manual_review,
        "enable_rag": enable_rag,
        "target_variable": target_variable,
        "report_format": report_format
    }
    
    # 测试模式：仅用少量数据
    if test_mode:
        print("🧪 测试模式：将优先使用小样本进行快速验证")
        config["test_mode"] = True
    
    pipeline.configure_pipeline(**config)
    
    # 运行分析
    start_time = datetime.now()
    
    try:
        results = pipeline.run_complete_pipeline()
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        print(f"\n🎉 分析完成！")
        print(f"⏱️ 总耗时: {duration}")
        print(f"📊 成功率: {results.get('success_rate', 0):.1%}")
        print(f"📄 报告路径: {results.get('report_path', 'N/A')}")
        
        # 显示结果摘要
        print(f"\n📈 执行摘要:")
        for step, result in results.get('results', {}).items():
            status = result.get('status', 'unknown')
            emoji = "✅" if status == "completed" else "❌"
            print(f"  {emoji} {step}: {status}")
        
        return results
        
    except Exception as e:
        print(f"❌ 分析过程中出现错误: {str(e)}")
        return None

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="数据科学智能分析系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python run_analysis.py data.csv                          # 基础分析
  python run_analysis.py data.csv --format html            # HTML格式报告
  python run_analysis.py data.csv --format pdf             # PDF格式报告
  python run_analysis.py data.csv --target price --test    # 测试模式
  python run_analysis.py data.csv --manual --rag           # 完整交互模式
        """
    )
    
    parser.add_argument("data_path", help="数据文件路径")
    parser.add_argument("--target", "-t", help="目标变量名称")
    parser.add_argument("--format", "-f", 
                        choices=["markdown", "html", "json", "pdf"],
                        default="markdown",
                        help="报告输出格式 (默认: markdown)")
    parser.add_argument("--manual", "-m", action="store_true",
                        help="启用人工审核模式")
    parser.add_argument("--rag", "-r", action="store_true",
                        help="启用RAG功能")
    parser.add_argument("--test", action="store_true",
                        help="启用测试模式（小样本快速验证）")
    parser.add_argument("--skip-check", action="store_true",
                        help="跳过系统要求检查")
    
    args = parser.parse_args()
    
    # 系统要求检查
    if not args.skip_check:
        if not check_requirements():
            sys.exit(1)
    
    # 设置目录
    setup_directories()
    
    # 检查数据文件
    if not os.path.exists(args.data_path):
        print(f"❌ 数据文件不存在: {args.data_path}")
        sys.exit(1)
    
    # 运行分析
    results = run_analysis(
        data_path=args.data_path,
        target_variable=args.target,
        report_format=args.format,
        enable_manual_review=args.manual,
        enable_rag=args.rag,
        test_mode=args.test
    )
    
    if results:
        print(f"\n✨ 分析完成！查看报告了解详细结果。")
        if args.format == "pdf":
            print(f"💡 提示: 如果PDF生成失败，请安装: pip install weasyprint markdown")
    else:
        print(f"\n❌ 分析失败，请检查错误信息并重试。")
        sys.exit(1)

if __name__ == "__main__":
    main() 