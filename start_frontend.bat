@echo off
echo 正在检查端口8101是否被占用...

:: 查找占用8101端口的进程PID
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8101 ^| findstr LISTENING') do (
    set pid=%%a
    echo 发现端口8101被进程PID: %%a 占用
    
    :: 终止该进程
    echo 正在终止进程 PID: %%a
    taskkill /F /PID %%a
    echo 端口8101已释放
)

echo 启动PropertyWize前端应用...
cd frontend
npm run start

pause 