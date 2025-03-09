#!/usr/bin/env python
"""
检查前端和后端服务是否正常运行
"""
import requests
import time
import socket
import sys

def check_port(host, port):
    """检查指定端口是否可以连接"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(2)
    result = sock.connect_ex((host, port))
    sock.close()
    return bool(result == 0)

def check_service(url, service_name):
    """检查服务是否正常响应"""
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            print(f"✅ {service_name}服务正常运行 - 状态码: {response.status_code}")
            return True
        else:
            print(f"❌ {service_name}服务返回了非200状态码: {response.status_code}")
            return False
    except requests.RequestException as e:
        print(f"❌ 无法连接到{service_name}服务: {str(e)}")
        return False

def main():
    """主函数"""
    print("=" * 50)
    print("服务连接检查工具")
    print("=" * 50)
    
    # 检查端口
    print("\n检查端口状态...")
    frontend_port_open = check_port("localhost", 3001)
    backend_port_open = check_port("localhost", 8000)
    
    if frontend_port_open:
        print("✅ 前端端口 (3001) 已开放")
    else:
        print("❌ 前端端口 (3001) 未开放 - 请确认前端服务已启动")
    
    if backend_port_open:
        print("✅ 后端端口 (8000) 已开放")
    else:
        print("❌ 后端端口 (8000) 未开放 - 请确认后端服务已启动")
    
    # 如果端口开放，检查服务
    if frontend_port_open:
        print("\n检查前端服务...")
        check_service("http://localhost:3001", "前端")
    
    if backend_port_open:
        print("\n检查后端服务...")
        check_service("http://localhost:8000", "后端")
        check_service("http://localhost:8000/api/health", "后端健康检查")
    
    print("\n诊断建议:")
    if not frontend_port_open:
        print("1. 重新启动前端服务: cd frontend && npm start")
    if not backend_port_open:
        print("1. 重新启动后端服务: cd backend && python main.py")
    
    print("\n其他可能的解决方案:")
    print("1. 尝试使用不同的端口: 在 vite.config.ts 中更改端口")
    print("2. 检查防火墙是否阻止了端口访问")
    print("3. 确保没有其他程序占用这些端口")
    print("4. 尝试重启电脑")
    
    print("\n如果所有服务都正常但仍无法访问，请尝试:")
    print("1. 直接访问: http://localhost:3001/#/")
    print("2. 或者通过IP地址: http://127.0.0.1:3001/")
    
if __name__ == "__main__":
    main() 