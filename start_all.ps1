# PropertyWize全栈应用启动脚本

Write-Host "正在启动PropertyWize全栈应用..." -ForegroundColor Cyan

# 检查并释放前端端口(3001)
Write-Host "检查前端端口3001是否被占用..." -ForegroundColor Yellow
$frontendPort = 3001
$frontendProcesses = Get-NetTCPConnection -LocalPort $frontendPort -ErrorAction SilentlyContinue | Where-Object State -eq Listen

if ($frontendProcesses) {
    foreach ($process in $frontendProcesses) {
        $processId = $process.OwningProcess
        $processName = (Get-Process -Id $processId).ProcessName
        Write-Host "发现端口$frontendPort被进程PID: $processId ($processName) 占用" -ForegroundColor Red
        Write-Host "正在终止进程 PID: $processId" -ForegroundColor Red
        Stop-Process -Id $processId -Force
        Write-Host "端口$frontendPort已释放" -ForegroundColor Green
    }
}

# 检查并释放后端端口(8000)
Write-Host "检查后端端口8000是否被占用..." -ForegroundColor Yellow
$backendPort = 8000
$backendProcesses = Get-NetTCPConnection -LocalPort $backendPort -ErrorAction SilentlyContinue | Where-Object State -eq Listen

if ($backendProcesses) {
    foreach ($process in $backendProcesses) {
        $processId = $process.OwningProcess
        $processName = (Get-Process -Id $processId).ProcessName
        Write-Host "发现端口$backendPort被进程PID: $processId ($processName) 占用" -ForegroundColor Red
        Write-Host "正在终止进程 PID: $processId" -ForegroundColor Red
        Stop-Process -Id $processId -Force
        Write-Host "端口$backendPort已释放" -ForegroundColor Green
    }
}

# 启动后端API服务
Write-Host "正在启动后端API服务 - 端口8000" -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PSScriptRoot\backend'; python main.py"

# 等待几秒让后端启动
Write-Host "等待后端服务启动..." -ForegroundColor Yellow
Start-Sleep -Seconds 3

# 启动前端应用
Write-Host "正在启动前端应用 - 端口3001" -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PSScriptRoot\frontend'; npm run dev"

Write-Host "PropertyWize全栈应用已启动！" -ForegroundColor Green
Write-Host "前端: http://localhost:3001" -ForegroundColor Green
Write-Host "后端API: http://localhost:8000" -ForegroundColor Green