"""Модели данных"""
from dataclasses import dataclass


@dataclass
class SubtitleBlock:
    """Блок субтитров"""
    number: str
    timestamp: str
    text: str
    
    def __str__(self) -> str:
        return f"[{self.number}] {self.timestamp}\n{self.text}"