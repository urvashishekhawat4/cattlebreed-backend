from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
import os

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=True)
    email = Column(String(120), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    profile_picture_url = Column(String(255), nullable=True)

class PredictionLog(Base):
    __tablename__ = 'prediction_logs'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    image_filename = Column(String(255), nullable=False)
    breed_result = Column(String(100), nullable=False)
    breed_confidence = Column(Float, nullable=False)
    bcs_result = Column(String(100), nullable=False)
    bcs_confidence = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)

# Ensure database is in the backend directory
db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "database.db"))
engine = create_engine(f'sqlite:///{db_path}')
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    Base.metadata.create_all(bind=engine)
