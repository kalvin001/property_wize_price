from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File, Form, Depends
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
import tempfile
import json
import shutil
import subprocess
import time
from datetime import datetime
from database.project_manager import project_manager
from database.models import DeploymentRecord

# API路由
router = APIRouter(prefix="/api/projects", tags=["项目管理"])

# 数据模型
class ProjectCreate(BaseModel):
    name: str
    description: Optional[str] = None

class ProjectUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None

class ProjectScan(BaseModel):
    project_id: Optional[int] = None
    name: Optional[str] = None

class ProjectDeploy(BaseModel):
    environment: str
    build_frontend: bool = True
    auto_restart: bool = True
    
class ProjectResponse(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    config: Optional[Dict[str, Any]] = None
    deployments: Optional[List[Dict[str, Any]]] = None

# API接口
@router.get("/", response_model=List[ProjectResponse])
async def get_all_projects():
    """获取所有项目"""
    projects = project_manager.get_all_projects()
    return projects

@router.post("/", response_model=ProjectResponse)
async def create_project(project_data: ProjectCreate):
    """创建新项目"""
    project = project_manager.create_project(
        name=project_data.name,
        description=project_data.description
    )
    return project

@router.get("/{project_id}", response_model=ProjectResponse)
async def get_project(project_id: int):
    """获取项目详情"""
    project = project_manager.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail=f"未找到ID为{project_id}的项目")
    
    # 获取项目的部署记录
    session = project_manager.db.get_session()
    try:
        deployments = session.query(DeploymentRecord).filter(
            DeploymentRecord.project_id == project_id
        ).order_by(DeploymentRecord.created_at.desc()).all()
        
        # 转换为可序列化的格式
        deployment_list = []
        for deployment in deployments:
            deployment_list.append({
                "id": deployment.id,
                "environment": deployment.environment,
                "status": deployment.status,
                "details": deployment.details,
                "created_at": deployment.created_at
            })
        
        # 为了保持与现有API兼容，我们将部署记录添加到项目对象中
        setattr(project, "deployments", deployment_list)
    finally:
        session.close()
    
    return project

@router.put("/{project_id}", response_model=ProjectResponse)
async def update_project(project_id: int, project_data: ProjectUpdate):
    """更新项目"""
    update_data = project_data.dict(exclude_unset=True)
    project = project_manager.update_project(project_id, **update_data)
    if not project:
        raise HTTPException(status_code=404, detail=f"未找到ID为{project_id}的项目")
    return project

@router.delete("/{project_id}")
async def delete_project(project_id: int):
    """删除项目"""
    success = project_manager.delete_project(project_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"未找到ID为{project_id}的项目")
    return {"message": f"项目ID={project_id}已成功删除"}

@router.post("/scan", response_model=ProjectResponse)
async def scan_project(scan_data: ProjectScan, background_tasks: BackgroundTasks):
    """扫描项目结构"""
    try:
        project = project_manager.scan_project_structure(
            project_name=scan_data.name,
            project_id=scan_data.project_id
        )
        return project
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"扫描项目时发生错误: {str(e)}")

@router.post("/{project_id}/update", response_model=ProjectResponse)
async def update_project_from_repo(project_id: int, background_tasks: BackgroundTasks):
    """从代码仓库更新项目"""
    try:
        # 先获取项目
        project = project_manager.get_project(project_id)
        if not project:
            raise HTTPException(status_code=404, detail=f"未找到ID为{project_id}的项目")
        
        # 检查是否为Git仓库
        if not os.path.exists(".git"):
            raise HTTPException(status_code=400, detail="当前目录不是Git仓库")
        
        # 执行git pull
        result = subprocess.run(
            ["git", "pull"], 
            capture_output=True, 
            text=True
        )
        
        if result.returncode != 0:
            raise HTTPException(
                status_code=500, 
                detail=f"Git更新失败: {result.stderr}"
            )
        
        # 更新项目的配置信息
        project = project_manager.scan_project_structure(project_id=project_id)
        
        # 记录更新操作
        session = project_manager.db.get_session()
        try:
            deployment = DeploymentRecord(
                project_id=project_id,
                environment="update",
                status="success",
                details={
                    "message": "项目代码已从仓库更新",
                    "git_output": result.stdout
                }
            )
            session.add(deployment)
            session.commit()
        finally:
            session.close()
        
        return project
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"更新项目时发生错误: {str(e)}")

@router.post("/{project_id}/deploy")
async def deploy_project(
    project_id: int, 
    deploy_data: ProjectDeploy,
    background_tasks: BackgroundTasks
):
    """部署项目"""
    try:
        # 先获取项目
        project = project_manager.get_project(project_id)
        if not project:
            raise HTTPException(status_code=404, detail=f"未找到ID为{project_id}的项目")
        
        session = project_manager.db.get_session()
        try:
            # 创建部署记录
            deployment = DeploymentRecord(
                project_id=project_id,
                environment=deploy_data.environment,
                status="running",
                details={
                    "started_at": datetime.utcnow().isoformat(),
                    "config": deploy_data.dict()
                }
            )
            session.add(deployment)
            session.commit()
            session.refresh(deployment)
            
            # 注册部署任务
            background_tasks.add_task(
                _run_deployment,
                deployment_id=deployment.id,
                project_id=project_id,
                environment=deploy_data.environment,
                build_frontend=deploy_data.build_frontend,
                auto_restart=deploy_data.auto_restart
            )
            
            return {
                "message": "部署任务已开始",
                "deployment_id": deployment.id
            }
        finally:
            session.close()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"部署项目时发生错误: {str(e)}")

async def _run_deployment(
    deployment_id: int,
    project_id: int,
    environment: str,
    build_frontend: bool,
    auto_restart: bool
):
    """运行部署任务"""
    session = project_manager.db.get_session()
    try:
        # 获取部署记录
        deployment = session.query(DeploymentRecord).filter(
            DeploymentRecord.id == deployment_id
        ).first()
        
        if not deployment:
            print(f"无法找到部署记录: {deployment_id}")
            return
        
        steps = []
        success = True
        error_message = ""
        
        try:
            # 1. 构建前端
            if build_frontend and os.path.exists("frontend"):
                steps.append("构建前端")
                # 切换到前端目录
                os.chdir("frontend")
                
                # 安装依赖
                npm_install = subprocess.run(
                    ["npm", "install", "--legacy-peer-deps"], 
                    capture_output=True, 
                    text=True
                )
                if npm_install.returncode != 0:
                    raise Exception(f"安装前端依赖失败: {npm_install.stderr}")
                
                # 执行构建
                npm_build = subprocess.run(
                    ["npm", "run", "build"], 
                    capture_output=True, 
                    text=True
                )
                if npm_build.returncode != 0:
                    raise Exception(f"构建前端失败: {npm_build.stderr}")
                
                # 返回到项目根目录
                os.chdir("..")
            
            # 2. 安装后端依赖
            if os.path.exists("requirements.txt"):
                steps.append("安装后端依赖")
                pip_install = subprocess.run(
                    ["pip", "install", "-r", "requirements.txt"], 
                    capture_output=True, 
                    text=True
                )
                if pip_install.returncode != 0:
                    raise Exception(f"安装后端依赖失败: {pip_install.stderr}")
            
            # 3. 重启服务（如果需要）
            if auto_restart:
                steps.append("重启服务")
                if os.name == 'nt':  # Windows
                    # 在Windows中，我们可能需要使用不同的方式重启服务
                    if os.path.exists("start_all.bat"):
                        restart = subprocess.run(
                            ["start_all.bat"], 
                            capture_output=True, 
                            text=True
                        )
                    elif os.path.exists("start_all.ps1"):
                        restart = subprocess.run(
                            ["powershell", "-ExecutionPolicy", "Bypass", "-File", "start_all.ps1"], 
                            capture_output=True, 
                            text=True
                        )
                    else:
                        raise Exception("未找到重启脚本")
                else:  # Unix/Linux
                    if os.path.exists("start_all.sh"):
                        restart = subprocess.run(
                            ["bash", "start_all.sh"], 
                            capture_output=True, 
                            text=True
                        )
                    else:
                        raise Exception("未找到重启脚本")
                
                if restart.returncode != 0:
                    raise Exception(f"重启服务失败: {restart.stderr}")
            
            # 4. 更新部署记录
            deployment.status = "success"
            deployment.details.update({
                "completed_at": datetime.utcnow().isoformat(),
                "steps": steps,
                "success": True
            })
            
        except Exception as e:
            success = False
            error_message = str(e)
            
            # 更新部署记录为失败
            deployment.status = "failed"
            deployment.details.update({
                "completed_at": datetime.utcnow().isoformat(),
                "steps": steps,
                "success": False,
                "error": error_message
            })
        
        session.commit()
        
    except Exception as e:
        print(f"部署过程中发生错误: {e}")
    finally:
        session.close()

@router.get("/{project_id}/export")
async def export_project(project_id: int, background_tasks: BackgroundTasks):
    """导出项目为ZIP文件"""
    try:
        # 创建临时目录用于导出
        temp_dir = tempfile.mkdtemp()
        output_path = os.path.join(temp_dir, f"project_{project_id}_export.zip")
        
        # 执行导出
        project_manager.export_project(project_id, output_path)
        
        # 设置清理函数
        def cleanup():
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                
        # 添加到后台任务，在响应发送后执行清理
        background_tasks.add_task(cleanup)
        
        # 返回文件
        return FileResponse(
            path=output_path,
            filename=f"project_{project_id}_export.zip",
            media_type="application/zip"
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"导出项目时发生错误: {str(e)}")

@router.post("/import")
async def import_project(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """从ZIP文件导入项目"""
    try:
        # 创建临时目录
        temp_dir = tempfile.mkdtemp()
        temp_file = os.path.join(temp_dir, "import.zip")
        
        # 保存上传的文件
        with open(temp_file, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # 导入项目
        project = project_manager.import_project(temp_file)
        
        # 设置清理函数
        def cleanup():
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                
        # 添加到后台任务，在响应发送后执行清理
        background_tasks.add_task(cleanup)
        
        return {
            "message": "项目导入成功",
            "project_id": project.id,
            "project_name": project.name
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"导入项目时发生错误: {str(e)}")

# 添加到主应用中
def init_project_api(app):
    """初始化项目管理API"""
    app.include_router(router) 