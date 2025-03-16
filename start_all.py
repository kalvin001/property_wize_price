#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import subprocess
import time
import socket
import signal
import platform

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

def start_backend(max_retries=3):
    """启动后端API服务，支持重试"""
    print("正在启动后端API服务 - 端口8000")
    
    # 启动后端
    backend_cmd = 'start cmd /k "cd backend && python main.py"'
    subprocess.Popen(backend_cmd, shell=True)
    
    # 检查后端是否成功启动
    retry_count = 0
    while retry_count < max_retries:
        print(f"等待后端启动 (尝试 {retry_count+1}/{max_retries})...")
        time.sleep(3)
        
        if check_port_in_use(8000):
            print("后端API服务已成功启动")
            return True
        
        retry_count += 1
        
        if retry_count < max_retries:
            print("后端启动可能失败，正在重试...")
            # 关闭可能存在的失败进程
            find_and_kill_process_by_port(8000)
            # 重新启动
            subprocess.Popen(backend_cmd, shell=True)
    
    print("警告: 后端启动多次尝试后仍然失败")
    return False

def start_frontend():
    """启动前端应用"""
    print("正在启动前端应用 - 端口3001")
    frontend_cmd = 'start cmd /k "cd frontend && npm run dev"'
    subprocess.Popen(frontend_cmd, shell=True)
    
    # 等待前端启动
    time.sleep(2)
    print("前端应用已启动")

def start_application():
    """启动PropertyWize全栈应用"""
    print("正在启动PropertyWize全栈应用...")

    # 检查并释放前端和后端端口
    find_and_kill_process_by_port(3001)  # 前端端口
    find_and_kill_process_by_port(8000)  # 后端端口

    # 启动后端API服务
    backend_started = start_backend()
    
    # 无论后端是否成功启动，都尝试启动前端
    start_frontend()

    print("PropertyWize全栈应用已启动！")
    print("前端: http://localhost:3001")
    print("后端API: http://localhost:8000")
    
    if not backend_started:
        print("\n警告: 后端可能未正确启动，请检查后端控制台输出并手动重启后端")
        print("如果出现错误，可能需要修复main.py中的lifespan实现")

if __name__ == "__main__":
    start_application() 