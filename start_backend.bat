@echo off
echo 正在检查端口8000是否被占用...

:: 查找占用8000端口的进程PID
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8000 ^| findstr LISTENING') do (
    set pid=%%a
    echo 发现端口8000被进程PID: %%a 占用
    
    :: 终止该进程
    echo 正在终止进程 PID: %%a
    taskkill /F /PID %%a
    echo 端口8000已释放
)

echo 启动后端API服务 - 端口8000
cd backend
python main.py 