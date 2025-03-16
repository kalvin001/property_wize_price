#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import subprocess
import time
import socket
import signal
import platform
import argparse
import sys
import datetime
from pathlib import Path

# 检测当前操作系统
IS_WINDOWS = platform.system().lower() == 'windows'

# 确定Python命令
PYTHON_CMD = 'python' if IS_WINDOWS else 'python3'

def check_port_in_use(port):
    """检查指定端口是否被占用"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def find_and_kill_process_by_port(port):
    """查找并终止占用指定端口的进程"""
    print(f"检查端口{port}是否被占用...")
    
    if not check_port_in_use(port):
        print(f"端口{port}未被占用")
        return
    
    if IS_WINDOWS:
        # Windows系统下查找并终止进程
        try:
            output = subprocess.check_output(f'netstat -ano | findstr :{port} | findstr LISTENING', shell=True).decode('gbk')
            if output:
                pid = output.strip().split()[-1]
                print(f"发现端口{port}被进程PID: {pid} 占用")
                print(f"正在终止进程 PID: {pid}")
                os.system(f'taskkill /F /PID {pid}')
                print(f"端口{port}已释放")
        except subprocess.CalledProcessError:
            print(f"没有进程占用端口{port}")
    else:
        # Linux系统下查找并终止进程
        try:
            output = subprocess.check_output(f'lsof -i :{port} -t', shell=True).decode('utf-8')
            if output:
                pids = output.strip().split('\n')
                for pid in pids:
                    print(f"发现端口{port}被进程PID: {pid} 占用")
                    print(f"正在终止进程 PID: {pid}")
                    os.system(f'kill -9 {pid}')
                print(f"端口{port}已释放")
        except subprocess.CalledProcessError:
            print(f"没有进程占用端口{port}")

def start_backend(max_retries=3, background=False):
    """启动后端API服务，支持重试"""
    print("正在启动后端API服务 - 端口8102")
    
    logs_dir = Path("logs")
    log_file = str(logs_dir / f"backend_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    # 根据操作系统选择不同的启动方式
    if IS_WINDOWS:
        if background:
            backend_cmd = f'cd backend && start /min cmd /c "python main.py > {os.path.abspath(log_file)} 2>&1"'
            subprocess.Popen(backend_cmd, shell=True)
        else:
            backend_cmd = 'cd backend && start cmd /k "python main.py"'
            subprocess.Popen(backend_cmd, shell=True)
    else:
        # Linux环境
        backend_dir = os.path.join(os.getcwd(), 'backend')
        if background:
            with open(log_file, 'w') as f:
                process = subprocess.Popen(
                    [PYTHON_CMD, 'main.py'], 
                    cwd=backend_dir,
                    stdout=f, 
                    stderr=subprocess.STDOUT,
                    preexec_fn=os.setsid if hasattr(os, 'setsid') else None
                )
        else:
            process = subprocess.Popen(
                [PYTHON_CMD, 'main.py'],
                cwd=backend_dir
            )
    
    # 检查后端是否成功启动
    retry_count = 0
    while retry_count < max_retries:
        print(f"等待后端启动 (尝试 {retry_count+1}/{max_retries})...")
        time.sleep(3)
        
        if check_port_in_use(8102):
            print("后端API服务已成功启动")
            return True
        
        retry_count += 1
        
        if retry_count < max_retries:
            print("后端启动可能失败，正在重试...")
            # 关闭可能存在的失败进程
            find_and_kill_process_by_port(8102)
            # 重新启动
            if IS_WINDOWS:
                if background:
                    backend_cmd = f'cd backend && start /min cmd /c "python main.py > {os.path.abspath(log_file)} 2>&1"'
                    subprocess.Popen(backend_cmd, shell=True)
                else:
                    backend_cmd = 'cd backend && start cmd /k "python main.py"'
                    subprocess.Popen(backend_cmd, shell=True)
            else:
                # Linux环境
                backend_dir = os.path.join(os.getcwd(), 'backend')
                if background:
                    with open(log_file, 'w') as f:
                        process = subprocess.Popen(
                            [PYTHON_CMD, 'main.py'], 
                            cwd=backend_dir,
                            stdout=f, 
                            stderr=subprocess.STDOUT,
                            preexec_fn=os.setsid if hasattr(os, 'setsid') else None
                        )
                else:
                    process = subprocess.Popen(
                        [PYTHON_CMD, 'main.py'],
                        cwd=backend_dir
                    )
    
    print("警告: 后端启动多次尝试后仍然失败")
    return False

def start_frontend(background=False):
    """启动前端应用"""
    print("正在启动前端应用 - 端口8101")
    
    logs_dir = Path("logs")
    log_file = str(logs_dir / f"frontend_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    # 根据操作系统选择不同的启动方式
    if IS_WINDOWS:
        if background:
            frontend_cmd = f'cd frontend && start /min cmd /c "npm run dev > {os.path.abspath(log_file)} 2>&1"'
            subprocess.Popen(frontend_cmd, shell=True)
        else:
            frontend_cmd = 'cd frontend && start cmd /k "npm run dev"'
            subprocess.Popen(frontend_cmd, shell=True)
    else:
        # Linux环境
        frontend_dir = os.path.join(os.getcwd(), 'frontend')
        if background:
            with open(log_file, 'w') as f:
                process = subprocess.Popen(
                    ['npm', 'run', 'dev'], 
                    cwd=frontend_dir,
                    stdout=f, 
                    stderr=subprocess.STDOUT,
                    preexec_fn=os.setsid if hasattr(os, 'setsid') else None
                )
        else:
            process = subprocess.Popen(
                ['npm', 'run', 'dev'],
                cwd=frontend_dir
            )
    
    # 等待前端启动
    time.sleep(2)
    print("前端应用已启动")

def start_application(background=False):
    """启动PropertyWize全栈应用"""
    print("正在启动PropertyWize全栈应用...")

    # 确保logs目录存在
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    # 检查并释放前端和后端端口
    find_and_kill_process_by_port(8101)  # 前端端口
    find_and_kill_process_by_port(8102)  # 后端端口

    # 启动后端API服务
    backend_started = start_backend(background=background)
    
    # 无论后端是否成功启动，都尝试启动前端
    start_frontend(background=background)

    print("PropertyWize全栈应用已启动！")
    print("前端: http://localhost:8101")
    print("后端API: http://localhost:8102")
    
    if background:
        print(f"应用在后台运行中，日志保存在 {os.path.abspath('logs')} 目录下")
    
    if not backend_started:
        print("\n警告: 后端可能未正确启动，请检查后端控制台输出并手动重启后端")
        print("如果出现错误，可能需要修复main.py中的lifespan实现")

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='启动PropertyWize全栈应用')
    parser.add_argument('--background', '-b', action='store_true', help='在后台运行并将日志写入logs目录')
    args = parser.parse_args()
    
    start_application(background=args.background)

# Linux环境下可以使用以下命令在后台运行:
# nohup python3 start_all.py -b > start_all.log 2>&1 &