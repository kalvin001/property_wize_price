@echo off
echo 正在启动PropertyWize全栈应用...

:: 检查并释放前端端口(8101)
echo 检查前端端口8101是否被占用...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8101 ^| findstr LISTENING') do (
    set pid=%%a
    echo 发现端口8101被进程PID: %%a 占用
    echo 正在终止进程 PID: %%a
    taskkill /F /PID %%a
    echo 端口8101已释放
)

:: 检查并释放后端端口(8102)
echo 检查后端端口8102是否被占用...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8102 ^| findstr LISTENING') do (
    set pid=%%a
    echo 发现端口8102被进程PID: %%a 占用
    echo 正在终止进程 PID: %%a
    taskkill /F /PID %%a
    echo 端口8102已释放
)

:: 启动后端API服务
echo 正在启动后端API服务 - 端口8102
start cmd /k "cd backend && python main.py"

:: 等待几秒让后端启动
timeout /t 3

:: 启动前端应用
echo 正在启动前端应用 - 端口8101
start cmd /k "cd frontend && npm run dev"

echo PropertyWize全栈应用已启动！
echo 前端: http://localhost:8101
echo 后端API: http://localhost:8102

exit