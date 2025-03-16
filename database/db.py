from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from .models import Base
import os

class Database:
    def __init__(self, db_path="sqlite:///property_wize.db"):
        self.engine = create_engine(db_path)
        self.SessionLocal = sessionmaker(bind=self.engine)
        
    def create_tables(self):
        Base.metadata.create_all(self.engine)
        
    def get_session(self):
        return self.SessionLocal()
    
    def close(self):
        self.engine.dispose()

# 创建默认数据库实例
db = Database()

# 确保数据库表存在
db.create_tables() 