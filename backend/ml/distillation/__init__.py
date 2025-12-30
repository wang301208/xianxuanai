"""Knowledge distillation utilities."""

from .teacher_student_trainer import DistillationConfig, TeacherStudentTrainer
from .utils import fine_tune, progressive_knowledge_accumulation

__all__ = [
    "DistillationConfig",
    "TeacherStudentTrainer",
    "fine_tune",
    "progressive_knowledge_accumulation",
]
