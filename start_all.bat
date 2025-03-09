@echo off
echo 正在启动PropertyWize全栈应用...

:: 检查并释放前端端口(3001)
echo 检查前端端口3001是否被占用...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :3001 ^| findstr LISTENING') do (
    set pid=%%a
    echo 发现端口3001被进程PID: %%a 占用
    echo 正在终止进程 PID: %%a
    taskkill /F /PID %%a
    echo 端口3001已释放
)

:: 检查并释放后端端口(8000)
echo 检查后端端口8000是否被占用...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8000 ^| findstr LISTENING') do (
    set pid=%%a
    echo 发现端口8000被进程PID: %%a 占用
    echo 正在终止进程 PID: %%a
    taskkill /F /PID %%a
    echo 端口8000已释放
)

:: 启动后端API服务
echo 正在启动后端API服务 - 端口8000
start cmd /k "cd backend && python main.py"

:: 等待几秒让后端启动
timeout /t 3

:: 启动前端应用
echo 正在启动前端应用 - 端口3001
start cmd /k "cd frontend && npm run dev"

echo PropertyWize全栈应用已启动！
echo 前端: http://localhost:3001
echo 后端API: http://localhost:8000

exit