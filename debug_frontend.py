#!/usr/bin/env python
"""
前端服务调试工具
"""
import os
import subprocess
import time
import sys
import socket
import psutil

def check_port_in_use(port):
    """检查端口是否被占用"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return bool(s.connect_ex(('127.0.0.1', port)) == 0)

def kill_process_on_port(port):
    """终止占用指定端口的进程"""
    if not check_port_in_use(port):
        print(f"端口 {port} 未被占用")
        return False
    
    try:
        for proc in psutil.process_iter(['pid', 'name', 'connections']):
            connections = proc.info.get('connections', [])
            for conn in connections:
                if conn.laddr.port == port:
                    print(f"发现进程使用端口 {port}: PID={proc.info['pid']}, 名称={proc.info['name']}")
                    proc.terminate()
                    print(f"已终止进程 PID={proc.info['pid']}")
                    return True
    except Exception as e:
        print(f"尝试终止进程时出错: {str(e)}")
    
    return False

def check_frontend_files():
    """检查前端文件是否完整"""
    frontend_dir = "frontend"
    if not os.path.exists(frontend_dir):
        print(f"错误: 前端目录 '{frontend_dir}' 不存在!")
        return False
    
    required_files = [
        "package.json",
        "vite.config.ts",
        "index.html",
        "src/main.tsx",
        "src/App.tsx",
        "src/HomePage.tsx"
    ]
    
    missing_files = []
    for file in required_files:
        full_path = os.path.join(frontend_dir, file)
        if not os.path.exists(full_path):
            missing_files.append(file)
    
    if missing_files:
        print(f"错误: 以下前端文件缺失:")
        for file in missing_files:
            print(f"  - {file}")
        return False
    
    print("前端文件检查通过")
    return True

def start_frontend_verbose():
    """详细模式启动前端，显示所有输出"""
    if not check_frontend_files():
        sys.exit(1)
    
    # 检查端口是否被占用
    port = 8101
    if check_port_in_use(port):
        print(f"警告: 端口 {port} 已被占用")
        choice = input(f"是否尝试释放端口 {port}? (y/n): ")
        if choice.lower() == 'y':
            if kill_process_on_port(port):
                print(f"成功释放端口 {port}")
            else:
                print(f"无法释放端口 {port}, 请手动关闭使用该端口的程序")
                
                # 尝试使用不同的端口
                port = 3002
                print(f"尝试使用端口 {port}")
    
    # 进入前端目录
    os.chdir("frontend")
    
    # 安装依赖
    if not os.path.exists("node_modules"):
        print("安装前端依赖...")
        npm_cmd = "npm.cmd" if os.name == 'nt' else "npm"
        subprocess.run([npm_cmd, "install"])
    
    # 启动前端服务
    print(f"启动前端服务于端口 {port}...")
    npm_cmd = "npm.cmd" if os.name == 'nt' else "npm"
    env = os.environ.copy()
    env["VITE_PORT"] = str(port)  # 设置环境变量
    
    try:
        # 使用实时输出模式
        process = subprocess.Popen(
            [npm_cmd, "run", "start"],
            env=env,
            shell=True
        )
        
        # 等待服务启动
        countdown = 10
        while countdown > 0:
            if check_port_in_use(port):
                print(f"前端服务成功启动! 访问 http://127.0.0.1:{port}/#/")
                return
            
            print(f"等待服务启动... {countdown}")
            time.sleep(1)
            countdown -= 1
        
        print("前端服务可能未成功启动，请检查上面的输出信息")
    except Exception as e:
        print(f"启动前端服务时出错: {str(e)}")

if __name__ == "__main__":
    print("=" * 50)
    print("前端服务调试工具")
    print("=" * 50)
    
    try:
        start_frontend_verbose()
        
        print("\n保持脚本运行中... 按 Ctrl+C 退出")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n脚本已停止")
    except Exception as e:
        print(f"发生错误: {str(e)}") 