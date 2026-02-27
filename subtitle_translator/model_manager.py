"""Управление загрузкой и инициализацией моделей"""
import os
import logging
import fasttext
from pathlib import Path
import warnings
import requests
from requests.exceptions import ConnectionError, Timeout

from subtitle_translator.downloader import ModelDownloader

logger = logging.getLogger(__name__)


class ModelManager:
    """Управление загрузкой и инициализацией моделей"""

    def __init__(self, config):
        self.config = config
        self.downloader = ModelDownloader(config)
        self.lang_detector = None
        self.ollama_available = False        # флаг доступности Ollama

    def load_lang_detector(self) -> bool:
        """Загружает модель определения языка"""
        if not self.downloader.download_file(
            self.config.fasttext_model_url,
            self.config.fasttext_model_path,
            self.config.fasttext_expected_size_mb
        ):
            return False

        try:
            self.lang_detector = fasttext.load_model(str(self.config.fasttext_model_path))
            logger.info("✓ Модель определения языка загружена")
            return True
        except Exception as e:
            logger.error(f"Ошибка загрузки модели определения языка: {e}")
            return False

    def load_translation_model(self) -> bool:
        """
        Проверяет доступность Ollama и наличие модели.
        Не загружает модель внутрь приложения, только проверяет связь.
        """
        # Проверяем, доступен ли сервер Ollama
        try:
            response = requests.get(f"{self.config.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                # Проверяем, есть ли нужная модель в списке
                model_names = [m["name"] for m in models]
                if self.config.ollama_model in model_names:
                    logger.info(f"✓ Ollama доступна, модель '{self.config.ollama_model}' найдена")
                    self.ollama_available = True
                    return True
                else:
                    logger.warning(
                        f"Модель '{self.config.ollama_model}' не найдена в Ollama. "
                        f"Доступные модели: {', '.join(model_names) if model_names else 'нет'}. "
                        f"Выполните: ollama pull {self.config.ollama_model}"
                    )
                    # Всё равно вернём True, так как модель можно загрузить позже,
                    # но translators будет проверять ollama_available и выдавать ошибку при попытке перевода.
                    self.ollama_available = False
                    return True
            else:
                logger.error(f"Ollama вернул ошибку: {response.status_code}")
                self.ollama_available = False
                return False
        except ConnectionError:
            logger.error(
                f"Не удалось подключиться к Ollama по адресу {self.config.ollama_url}. "
                "Убедитесь, что Ollama запущена."
            )
            self.ollama_available = False
            return False
        except Timeout:
            logger.error("Таймаут при подключении к Ollama.")
            self.ollama_available = False
            return False
        except Exception as e:
            logger.error(f"Ошибка при проверке Ollama: {e}")
            self.ollama_available = False
            return False