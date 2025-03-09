"""
房产估价分析系统启动脚本
"""
import os
import subprocess
import time
import webbrowser
import sys
from pathlib import Path
import socket

def check_requirements():
    """检查是否安装了必要的依赖"""
    print("检查系统依赖...")
    
    # 检查目录结构
    for directory in ["backend", "frontend", "model", "frontend/public/data"]:
        os.makedirs(directory, exist_ok=True)
    
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
    if not Path("model/xgb_model.joblib").exists():
        print("正在分析数据并训练模型...")
        try:
            subprocess.run(["python", "analyze_data.py"])
        except Exception as e:
            print(f"数据分析过程中出错: {str(e)}")
            print("将尝试继续启动系统，但某些功能可能不可用。")
    else:
        print("模型已存在，跳过训练步骤...")

def start_servers():
    """启动前端和后端服务器"""
    # 启动后端
    print("启动后端服务...")
    if not os.path.exists("backend/main.py"):
        print("错误: 后端主文件 (backend/main.py) 不存在！")
        sys.exit(1)
        
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
    
    # 检查前端文件
    if not os.path.exists("package.json"):
        print("错误: 前端配置文件 (frontend/package.json) 不存在！")
        os.chdir("..")  # 恢复目录
        sys.exit(1)
    
    # 安装依赖（如果需要）
    if not os.path.exists("node_modules"):
        print("安装前端依赖...")
        npm_cmd = "npm.cmd" if os.name == 'nt' else "npm"
        try:
            subprocess.run([npm_cmd, "install"], check=True)
        except subprocess.CalledProcessError:
            print("错误: 无法安装前端依赖")
            os.chdir("..")
            sys.exit(1)
    
    # 修复Windows下npm命令的问题
    if os.name == 'nt':  # Windows系统
        npm_cmd = "npm.cmd"
    else:
        npm_cmd = "npm"
    
    # 在Windows上，设置shell=True以确保命令正确执行
    use_shell = os.name == 'nt'
    
    # 设置环境变量来确保Vite使用正确的端口
    env = os.environ.copy()
    frontend_port = 3001
    env["PORT"] = str(frontend_port)
    env["VITE_PORT"] = str(frontend_port)
    
    # 使用start命令在Windows上启动分离的进程
    if use_shell and os.name == 'nt':
        cmd = f"start cmd /k {npm_cmd} run start:dev"
        frontend_process = subprocess.Popen(
            cmd,
            shell=True,
            env=env
        )
    else:
        # 在非Windows系统上使用标准方法
        frontend_process = subprocess.Popen(
            [npm_cmd, "run", "start:dev"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env
        )
        
    os.chdir("..")
    
    # 等待前端服务启动
    print("等待前端服务启动...")
    max_attempts = 10
    attempts = 0
    started = False
    
    while attempts < max_attempts:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex(('127.0.0.1', frontend_port)) == 0:
                    started = True
                    break
        except:
            pass
            
        attempts += 1
        time.sleep(1)
        print(f"等待前端服务启动... {attempts}/{max_attempts}")
    
    if started:
        print(f"前端服务已启动在端口 {frontend_port}")
    else:
        print("警告: 前端服务可能未成功启动")
    
    # 打开浏览器
    time.sleep(1)
    print("在浏览器中打开应用...")
    # 使用哈希路由格式的URL (#/) 和127.0.0.1
    url = f"http://127.0.0.1:{frontend_port}/#/"
    webbrowser.open(url)
    
    return backend_process, frontend_process, frontend_port

def main():
    """主函数"""
    print("=" * 50)
    print("房产估价分析系统")
    print("=" * 50)
    
    try:
        check_requirements()
        run_data_analysis()
        backend_process, frontend_process, frontend_port = start_servers()
        
        print("\n系统已启动!")
        print(f"前端: http://127.0.0.1:{frontend_port}/#/")
        print(f"后端: http://127.0.0.1:8000")
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
    except Exception as e:
        print(f"\n启动系统时发生错误: {str(e)}")
        print("请检查上述错误信息并修复问题。")
        sys.exit(1)
'''
npm run dev 后台启动
nohup npm run dev & 
nohup python3 backend/main.py &

'''
if __name__ == "__main__":
    main() 