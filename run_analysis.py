"""
房产估价分析系统启动脚本
"""
import os
import subprocess
import time
import webbrowser
from pathlib import Path

def check_requirements():
    """检查是否安装了必要的依赖"""
    print("检查系统依赖...")
    
    # 检查Python依赖
    backend_reqs = Path("backend/requirements.txt")
    if backend_reqs.exists():
        subprocess.run(["pip", "install", "-r", str(backend_reqs)])
    
    # 检查前端依赖
    if Path("frontend/package.json").exists() and not Path("frontend/node_modules").exists():
        print("正在安装前端依赖...")
        os.chdir("frontend")
        subprocess.run(["npm", "install"])
        os.chdir("..")

def run_data_analysis():
    """运行数据分析和模型训练"""
    if not Path("model/rf_model.joblib").exists():
        print("正在分析数据并训练模型...")
        subprocess.run(["python", "analyze_data.py"])
    else:
        print("模型已存在，跳过训练步骤...")

def start_servers():
    """启动前端和后端服务器"""
    # 启动后端
    print("启动后端服务...")
    backend_process = subprocess.Popen(
        ["python", "backend/main.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # 给后端一些启动时间
    time.sleep(2)
    
    # 启动前端
    print("启动前端服务...")
    os.chdir("frontend")
    
    # 修复Windows下npm命令的问题
    if os.name == 'nt':  # Windows系统
        npm_cmd = "npm.cmd"
    else:
        npm_cmd = "npm"
        
    frontend_process = subprocess.Popen(
        [npm_cmd, "start"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    os.chdir("..")
    
    # 前端端口设为3001（因为3000可能被占用）
    frontend_port = 3001
    
    # 打开浏览器
    time.sleep(3)
    print("在浏览器中打开应用...")
    webbrowser.open(f"http://localhost:{frontend_port}")
    
    return backend_process, frontend_process, frontend_port

def main():
    """主函数"""
    print("=" * 50)
    print("房产估价分析系统")
    print("=" * 50)
    
    check_requirements()
    run_data_analysis()
    backend_process, frontend_process, frontend_port = start_servers()
    
    print("\n系统已启动!")
    print(f"前端: http://localhost:{frontend_port}")
    print("后端: http://localhost:8000")
    print("\n按 Ctrl+C 停止服务")
    
    try:
        # 保持脚本运行
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n关闭服务...")
        backend_process.terminate()
        frontend_process.terminate()
        print("服务已停止")

if __name__ == "__main__":
    main() 