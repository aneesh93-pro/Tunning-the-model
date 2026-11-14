"""SQLAlchemy PostgreSQL ORM schema for GPU worker deployments."""
from datetime import datetime
from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Workspace(Base):
    __tablename__ = "workspaces"
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class Dataset(Base):
    __tablename__ = "datasets"
    id = Column(String, primary_key=True)
    workspace_id = Column(String, ForeignKey("workspaces.id"), index=True)
    name = Column(String, nullable=False)
    file_type = Column(String, nullable=False)
    format = Column(String, nullable=False)
    row_count = Column(Integer, nullable=False)
    artifact_path = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class TrainingExperiment(Base):
    __tablename__ = "training_experiments"
    id = Column(String, primary_key=True)
    dataset_id = Column(String, ForeignKey("datasets.id"), index=True)
    model_id = Column(String, nullable=False)
    method = Column(String, nullable=False)
    status = Column(String, nullable=False, index=True)
    config_json = Column(JSON, nullable=False)
    final_train_loss = Column(Float, nullable=True)
    final_val_loss = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    metrics = relationship("TrainingMetric", back_populates="experiment")


class TrainingMetric(Base):
    __tablename__ = "training_metrics"
    id = Column(Integer, primary_key=True, autoincrement=True)
    experiment_id = Column(String, ForeignKey("training_experiments.id"), index=True)
    step = Column(Integer, nullable=False)
    epoch = Column(Integer, nullable=False)
    train_loss = Column(Float, nullable=False)
    val_loss = Column(Float, nullable=True)
    learning_rate = Column(Float, nullable=False)
    experiment = relationship("TrainingExperiment", back_populates="metrics")
