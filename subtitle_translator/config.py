"""Конфигурация приложения"""
from pathlib import Path
from dataclasses import dataclass
import os


@dataclass
class Config:
    """Конфигурация приложения"""
    models_dir: Path = Path("models")
    fasttext_model_url: str = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
    fasttext_model_name: str = "lid.176.bin"
    fasttext_expected_size_mb: int = 126

    # Параметры подключения к Ollama
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"        # или любая другая подходящая модель
    ollama_timeout: int = 250                 # таймаут на генерацию (сек)
    
    # Параметры многопоточности
    max_workers: int = 6                      # количество потоков для параллельной обработки
    translation_batch_size: int = 10           # размер батча для группового перевода
    
    llm_system_prompt: str = (
        "Ты профессиональный переводчик, специализирующийся на спортивных репортажах по регби. "
        "Твоя задача - точно переводить текст на русский язык, сохраняя спортивную терминологию и сленг. "
        "Переводи ТОЛЬКО сам текст, НЕ добавляй никаких комментариев, пояснений или лишних слов. "
        "Если в тексте есть специфические регбийные термины, используй их правильные русские эквиваленты."
    )

    def __post_init__(self):
        # Создаем директории с правильными правами
        self.models_dir = Path(self.models_dir).absolute()
        self.models_dir.mkdir(exist_ok=True, mode=0o755)

        self.fasttext_model_path = self.models_dir / self.fasttext_model_name