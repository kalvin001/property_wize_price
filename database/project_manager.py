import os
import json
import shutil
import importlib
import glob
from pathlib import Path
from .db import db
from .models import Project, Model, PropertyReport
import sqlite3
import zipfile
import datetime

class ProjectManager:
    """项目管理器，用于项目的存储、检索和部署"""
    
    def __init__(self, db_instance=None):
        self.db = db_instance or db
        
    def create_project(self, name, description="", config=None):
        """创建新项目
        
        Args:
            name: 项目名称
            description: 项目描述
            config: 项目配置（字典）
            
        Returns:
            Project: 创建的项目对象
        """
        session = self.db.get_session()
        try:
            project = Project(
                name=name,
                description=description,
                config=config or {}
            )
            session.add(project)
            session.commit()
            session.refresh(project)
            return project
        finally:
            session.close()
            
    def get_project(self, project_id):
        """获取项目
        
        Args:
            project_id: 项目ID
            
        Returns:
            Project: 项目对象或None
        """
        session = self.db.get_session()
        try:
            return session.query(Project).filter(Project.id == project_id).first()
        finally:
            session.close()
            
    def get_all_projects(self):
        """获取所有项目
        
        Returns:
            list: 项目列表
        """
        session = self.db.get_session()
        try:
            return session.query(Project).all()
        finally:
            session.close()
            
    def update_project(self, project_id, **kwargs):
        """更新项目
        
        Args:
            project_id: 项目ID
            **kwargs: 要更新的字段
            
        Returns:
            Project: 更新后的项目对象或None
        """
        session = self.db.get_session()
        try:
            project = session.query(Project).filter(Project.id == project_id).first()
            if not project:
                return None
                
            for key, value in kwargs.items():
                if hasattr(project, key):
                    setattr(project, key, value)
            
            project.updated_at = datetime.datetime.utcnow()
            session.commit()
            session.refresh(project)
            return project
        finally:
            session.close()
            
    def delete_project(self, project_id):
        """删除项目
        
        Args:
            project_id: 项目ID
            
        Returns:
            bool: 是否删除成功
        """
        session = self.db.get_session()
        try:
            project = session.query(Project).filter(Project.id == project_id).first()
            if not project:
                return False
                
            session.delete(project)
            session.commit()
            return True
        finally:
            session.close()
            
    def scan_project_structure(self, project_name=None, project_id=None):
        """扫描项目结构，将项目文件信息存入数据库
        
        Args:
            project_name: 项目名称
            project_id: 项目ID
            
        Returns:
            Project: 项目对象
        """
        # 如果提供了project_id，则获取已有项目
        if project_id:
            project = self.get_project(project_id)
            if not project:
                raise ValueError(f"未找到ID为{project_id}的项目")
        # 如果提供了project_name，则创建新项目或获取已有项目
        elif project_name:
            session = self.db.get_session()
            try:
                project = session.query(Project).filter(Project.name == project_name).first()
                if not project:
                    project = Project(
                        name=project_name,
                        description=f"从项目名称{project_name}自动创建",
                        config={}
                    )
                    session.add(project)
                    session.commit()
                    session.refresh(project)
            finally:
                session.close()
        else:
            raise ValueError("必须提供project_name或project_id参数")
        
        # 扫描项目结构并更新项目配置
        project_config = {
            "files": self._scan_files(),
            "structure": self._analyze_structure(),
            "last_scan": datetime.datetime.utcnow().isoformat()
        }
        
        return self.update_project(project.id, config=project_config)
    
    def _scan_files(self):
        """扫描项目文件
        
        Returns:
            dict: 文件信息字典
        """
        ignore_dirs = ['.git', 'node_modules', '__pycache__', '.next']
        result = []
        
        for root, dirs, files in os.walk('.'):
            # 跳过忽略的目录
            dirs[:] = [d for d in dirs if d not in ignore_dirs]
            
            for file in files:
                if file.endswith(('.py', '.js', '.jsx', '.ts', '.tsx', '.json', '.css', '.scss', '.html')):
                    file_path = os.path.join(root, file)
                    try:
                        file_size = os.path.getsize(file_path)
                        result.append({
                            "path": file_path,
                            "size": file_size,
                            "type": os.path.splitext(file)[1][1:],
                            "last_modified": datetime.datetime.fromtimestamp(
                                os.path.getmtime(file_path)
                            ).isoformat()
                        })
                    except Exception as e:
                        print(f"无法处理文件 {file_path}: {e}")
        
        return result
    
    def _analyze_structure(self):
        """分析项目结构
        
        Returns:
            dict: 项目结构信息
        """
        structure = {
            "frontend": self._check_directory_exists("frontend"),
            "backend": self._check_directory_exists("backend"),
            "database": self._check_directory_exists("database"),
            "models": self._check_directory_exists("models") or self._check_directory_exists("model"),
            "resources": self._check_directory_exists("resources"),
            "has_package_json": os.path.exists("package.json"),
            "has_requirements_txt": os.path.exists("requirements.txt"),
            "frontend_framework": self._detect_frontend_framework(),
            "backend_framework": self._detect_backend_framework()
        }
        return structure
    
    def _check_directory_exists(self, dir_name):
        """检查目录是否存在
        
        Args:
            dir_name: 目录名
            
        Returns:
            bool: 目录是否存在
        """
        return os.path.isdir(dir_name)
    
    def _detect_frontend_framework(self):
        """检测前端框架
        
        Returns:
            str: 前端框架名称
        """
        if not os.path.exists("package.json"):
            return "unknown"
        
        try:
            with open("package.json", "r") as f:
                package_data = json.load(f)
                dependencies = {**package_data.get("dependencies", {}), **package_data.get("devDependencies", {})}
                
                if "react" in dependencies:
                    if "next" in dependencies:
                        return "next.js"
                    return "react"
                elif "vue" in dependencies:
                    return "vue"
                elif "angular" in dependencies:
                    return "angular"
                else:
                    return "unknown"
        except Exception:
            return "unknown"
    
    def _detect_backend_framework(self):
        """检测后端框架
        
        Returns:
            str: 后端框架名称
        """
        if os.path.exists("requirements.txt"):
            try:
                with open("requirements.txt", "r") as f:
                    content = f.read().lower()
                    if "fastapi" in content:
                        return "fastapi"
                    elif "flask" in content:
                        return "flask"
                    elif "django" in content:
                        return "django"
            except Exception:
                pass
        
        # 检查backend目录中的Python文件
        if os.path.isdir("backend"):
            for file in glob.glob("backend/*.py"):
                try:
                    with open(file, "r") as f:
                        content = f.read().lower()
                        if "fastapi" in content:
                            return "fastapi"
                        elif "flask" in content:
                            return "flask"
                        elif "django" in content:
                            return "django"
                except Exception:
                    continue
        
        return "unknown"
    
    def export_project(self, project_id, output_path):
        """导出项目为ZIP文件
        
        Args:
            project_id: 项目ID
            output_path: 输出路径
            
        Returns:
            str: 输出文件路径
        """
        project = self.get_project(project_id)
        if not project:
            raise ValueError(f"未找到ID为{project_id}的项目")
        
        # 创建临时目录
        temp_dir = f"temp_export_{project.name}"
        os.makedirs(temp_dir, exist_ok=True)
        
        # 导出项目数据
        conn = sqlite3.connect(f"{temp_dir}/project.db")
        cursor = conn.cursor()
        
        # 创建表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS project (
            id INTEGER PRIMARY KEY,
            name TEXT,
            description TEXT,
            config TEXT,
            created_at TEXT,
            updated_at TEXT
        )
        ''')
        
        # 插入项目信息
        cursor.execute(
            "INSERT INTO project VALUES (?, ?, ?, ?, ?, ?)",
            (
                project.id,
                project.name,
                project.description,
                json.dumps(project.config),
                project.created_at.isoformat(),
                project.updated_at.isoformat()
            )
        )
        
        conn.commit()
        conn.close()
        
        # 创建ZIP文件
        if not output_path.endswith('.zip'):
            output_path += '.zip'
            
        with zipfile.ZipFile(output_path, 'w') as zipf:
            zipf.write(f"{temp_dir}/project.db", "project.db")
            
            # 添加项目文件
            if project.config and 'files' in project.config:
                for file_info in project.config['files']:
                    file_path = file_info['path']
                    if os.path.exists(file_path) and os.path.isfile(file_path):
                        zipf.write(file_path, file_path)
        
        # 清理临时目录
        shutil.rmtree(temp_dir)
        
        return output_path
    
    def import_project(self, zip_path):
        """从ZIP文件导入项目
        
        Args:
            zip_path: ZIP文件路径
            
        Returns:
            Project: 导入的项目对象
        """
        if not os.path.exists(zip_path):
            raise ValueError(f"文件不存在: {zip_path}")
            
        # 创建临时目录
        temp_dir = f"temp_import_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
        os.makedirs(temp_dir, exist_ok=True)
        
        try:
            # 解压ZIP文件
            with zipfile.ZipFile(zip_path, 'r') as zipf:
                zipf.extractall(temp_dir)
            
            # 读取项目数据
            db_path = os.path.join(temp_dir, "project.db")
            if not os.path.exists(db_path):
                raise ValueError("无效的项目ZIP文件: 缺少project.db")
                
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT id, name, description, config, created_at, updated_at FROM project")
            row = cursor.fetchone()
            
            if not row:
                raise ValueError("项目数据库中没有项目记录")
                
            project_id, name, description, config_json, created_at, updated_at = row
            config = json.loads(config_json)
            
            # 创建项目
            session = self.db.get_session()
            try:
                # 检查是否已存在同名项目
                existing = session.query(Project).filter(Project.name == name).first()
                if existing:
                    name = f"{name}_imported_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
                    
                project = Project(
                    name=name,
                    description=description,
                    config=config,
                    created_at=datetime.datetime.fromisoformat(created_at),
                    updated_at=datetime.datetime.fromisoformat(updated_at)
                )
                session.add(project)
                session.commit()
                session.refresh(project)
                
                # 复制项目文件
                if 'files' in config:
                    for file_info in config['files']:
                        src_path = os.path.join(temp_dir, file_info['path'])
                        if os.path.exists(src_path) and os.path.isfile(src_path):
                            dst_path = file_info['path']
                            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                            shutil.copy2(src_path, dst_path)
                
                return project
            finally:
                session.close()
                
        finally:
            # 清理临时目录
            shutil.rmtree(temp_dir)

# 创建默认项目管理器实例
project_manager = ProjectManager() 