@echo off
echo 正在检查端口8102是否被占用...

:: 查找占用8102端口的进程PID
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8102 ^| findstr LISTENING') do (
    set pid=%%a
    echo 发现端口8102被进程PID: %%a 占用
    
    :: 终止该进程
    echo 正在终止进程 PID: %%a
    taskkill /F /PID %%a
    echo 端口8102已释放
)

echo 启动后端API服务 - 端口8102
cd backend
python main.py 