@echo off
echo 正在检查端口3001是否被占用...

:: 查找占用3001端口的进程PID
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :3001 ^| findstr LISTENING') do (
    set pid=%%a
    echo 发现端口3001被进程PID: %%a 占用
    
    :: 终止该进程
    echo 正在终止进程 PID: %%a
    taskkill /F /PID %%a
    echo 端口3001已释放
)

echo 启动PropertyWize前端应用...
cd frontend
npm run start

pause 