"""Загрузчик моделей с проверкой целостности"""
import time
import urllib.request
from pathlib import Path
import logging
import warnings
import ssl
import os

# Подавляем конкретное предупреждение fasttext
warnings.filterwarnings("ignore", message="load_model does not return WordVectorModel or SupervisedModel")

logger = logging.getLogger(__name__)


class ModelDownloader:
    """Класс для загрузки моделей с проверкой целостности"""
    
    def __init__(self, config):
        self.config = config
    
    def download_file(self, url: str, destination: Path, expected_size_mb: int, max_retries: int = 3) -> bool:
        """
        Скачивает файл с проверкой размера и возможностью возобновления.
        
        Args:
            url: URL для скачивания
            destination: путь для сохранения
            expected_size_mb: ожидаемый размер в МБ
            max_retries: максимальное количество попыток
            
        Returns:
            bool: True если успешно, False в противном случае
        """
        # Создаем SSL контекст для Linux систем
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        # Проверяем существующий файл
        if destination.exists():
            actual_size = destination.stat().st_size / (1024 * 1024)
            if actual_size > expected_size_mb * 0.9:
                # Дополнительная проверка, что файл не поврежден
                try:
                    import fasttext
                    # Пробуем загрузить модель для проверки целостности
                    fasttext.load_model(str(destination))
                    logger.info(f"✓ Файл {destination.name} существует и корректен ({actual_size:.1f} МБ)")
                    return True
                except Exception as e:
                    logger.warning(
                        f"Файл {destination.name} поврежден (ошибка загрузки: {e}). "
                        f"Размер: {actual_size:.1f} МБ из {expected_size_mb} МБ. Скачиваю заново..."
                    )
                    destination.unlink()
            else:
                logger.warning(
                    f"Файл {destination.name} слишком маленький ({actual_size:.1f} МБ из {expected_size_mb} МБ). "
                    "Скачиваю заново..."
                )
                destination.unlink()
        
        # Скачиваем файл
        for attempt in range(max_retries):
            try:
                logger.info(f"Скачивание {destination.name} (попытка {attempt + 1}/{max_retries})...")
                
                def report_progress(block_num, block_size, total_size):
                    downloaded = block_num * block_size / (1024 * 1024)
                    if total_size > 0:
                        total = total_size / (1024 * 1024)
                        percent = min(100, downloaded * 100 / total)
                        print(f"\rПрогресс: {downloaded:.1f} МБ / {total:.1f} МБ ({percent:.1f}%)", end='')
                
                # Создаем opener с SSL контекстом для Linux
                opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ssl_context))
                urllib.request.install_opener(opener)
                
                urllib.request.urlretrieve(url, destination, reporthook=report_progress)
                print()
                
                # Проверяем размер
                actual_size = destination.stat().st_size / (1024 * 1024)
                if actual_size < expected_size_mb * 0.9:
                    raise Exception(f"Файл слишком маленький: {actual_size:.1f} МБ")
                
                # Проверяем, что файл можно загрузить
                try:
                    import fasttext
                    fasttext.load_model(str(destination))
                except Exception as e:
                    raise Exception(f"Файл поврежден: {e}")
                
                logger.info(f"✓ {destination.name} успешно загружен и проверен ({actual_size:.1f} МБ)")
                return True
                
            except Exception as e:
                logger.error(f"Ошибка при скачивании: {e}")
                if attempt < max_retries - 1:
                    wait_time = 5 * (attempt + 1)
                    logger.info(f"Повторная попытка через {wait_time} секунд...")
                    time.sleep(wait_time)
                    if destination.exists():
                        destination.unlink()
                else:
                    logger.error(f"Не удалось скачать {destination.name} после {max_retries} попыток.")
                    return False
        
        return False