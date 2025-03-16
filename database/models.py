from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, JSON, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
import datetime

Base = declarative_base()

class Project(Base):
    __tablename__ = 'projects'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    description = Column(Text)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    config = Column(JSON)  # 存储项目配置
    
class Model(Base):
    __tablename__ = 'models'
    
    id = Column(Integer, primary_key=True)
    project_id = Column(Integer, ForeignKey('projects.id'))
    name = Column(String(100), nullable=False)
    version = Column(String(50))
    path = Column(String(255))  # 模型文件路径
    metrics = Column(JSON)  # 模型评估指标
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    project = relationship("Project", back_populates="models")

class PropertyReport(Base):
    __tablename__ = 'property_reports'
    
    id = Column(Integer, primary_key=True)
    prop_id = Column(String(100), nullable=False)
    std_address = Column(String(255))
    estimated_price = Column(Float)
    report_data = Column(JSON)  # 存储完整的报告数据
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    model_id = Column(Integer, ForeignKey('models.id'))
    
    model = relationship("Model")

class ProjectFile(Base):
    __tablename__ = 'project_files'
    
    id = Column(Integer, primary_key=True)
    project_id = Column(Integer, ForeignKey('projects.id'))
    path = Column(String(255), nullable=False)
    file_type = Column(String(50))  # 文件类型 (py, js, json, etc)
    size = Column(Integer)  # 文件大小（字节）
    content = Column(Text)  # 用于备份关键文件内容
    is_key_file = Column(Boolean, default=False)  # 是否是关键文件
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    
    project = relationship("Project", back_populates="files")

class Dependency(Base):
    __tablename__ = 'dependencies'
    
    id = Column(Integer, primary_key=True)
    project_id = Column(Integer, ForeignKey('projects.id'))
    name = Column(String(100), nullable=False)
    version = Column(String(50))
    dependency_type = Column(String(50))  # python, node, etc.
    is_dev = Column(Boolean, default=False)  # 是否是开发依赖
    
    project = relationship("Project", back_populates="dependencies")

class DeploymentRecord(Base):
    __tablename__ = 'deployment_records'
    
    id = Column(Integer, primary_key=True)
    project_id = Column(Integer, ForeignKey('projects.id'))
    environment = Column(String(50))  # 部署环境 (development, production)
    status = Column(String(50))  # 部署状态 (success, failed)
    details = Column(JSON)  # 部署详情
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    project = relationship("Project", back_populates="deployments")

# 添加反向关系
Project.models = relationship("Model", back_populates="project")
Project.files = relationship("ProjectFile", back_populates="project")
Project.dependencies = relationship("Dependency", back_populates="project")
Project.deployments = relationship("DeploymentRecord", back_populates="project") 