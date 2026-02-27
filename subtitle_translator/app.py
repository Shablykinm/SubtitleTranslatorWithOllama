"""Главное приложение"""
import logging
import datetime
import time
from pathlib import Path

from subtitle_translator.config import Config
from subtitle_translator.model_manager import ModelManager
from subtitle_translator.subtitle_processor import SubtitleProcessor
from subtitle_translator.translators import TranslationPipeline

logger = logging.getLogger(__name__)


class SubtitleTranslatorApp:
    """Главное приложение для перевода субтитров"""
    
    def __init__(self):
        self.config = Config()
        self.model_manager = ModelManager(self.config)
        self.pipeline = None
        self.start_time = None
    
    def initialize(self) -> bool:
        """Инициализация моделей"""
        logger.info("Инициализация переводчика субтитров")
        
        # Загружаем определитель языка
        if not self.model_manager.load_lang_detector():
            logger.error("Не удалось загрузить модель определения языка")
            return False
        
        # Загружаем модель перевода
        if not self.model_manager.load_translation_model():
            logger.error("Не удалось загрузить модель перевода")
            return False
        
        self.pipeline = TranslationPipeline(self.model_manager, self.config)
        return True
    
    def _format_time(self, seconds: float) -> str:
        """Форматирует время в ЧЧ:ММ:СС"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        else:
            return f"{minutes:02d}:{seconds:02d}"
    
    def run(self, input_file: str) -> bool:
        """
        Запуск обработки файла
        
        Args:
            input_file: путь к входному файлу субтитров
            
        Returns:
            bool: True если успешно, False в противном случае
        """
        self.start_time = time.time()
        
        # Проверяем входной файл
        input_path = Path(input_file)
        if not input_path.exists():
            logger.error(f"Файл {input_file} не найден")
            return False
        
        # Парсим субтитры
        logger.info(f"Парсинг субтитров из {input_file}...")
        try:
            blocks = SubtitleProcessor.parse_file(input_file)
            logger.info(f"✓ Найдено {len(blocks)} блоков субтитров")
        except Exception as e:
            logger.error(f"Ошибка при парсинге файла: {e}")
            return False
        
        if not blocks:
            logger.warning("Файл не содержит блоков субтитров")
            return False
        
        # Определяем исходный язык (только для информации)
        sample_blocks = blocks[:10] if len(blocks) > 10 else blocks
        sample_text = " ".join([b.text for b in sample_blocks if b.text.strip()])
        if sample_text:
            detected_lang = self.pipeline.lang_detector.detect(sample_text)
            logger.info(f"Определен исходный язык субтитров: {detected_lang}")
        
        # Переводим на русский язык с использованием контекста и многопоточности
        logger.info("=" * 60)
        logger.info("НАЧАЛО ПЕРЕВОДА")
        logger.info("=" * 60)
        
        try:
            # Выбираем метод перевода в зависимости от размера файла
            translation_start = time.time()
            
            if len(blocks) < 50:
                # Для маленьких файлов используем параллельную обработку блоков
                translated_blocks = self.pipeline.process_all_parallel(blocks)
            else:
                # Для больших файлов используем контекстный перевод с параллельной обработкой групп
                translated_blocks = self.pipeline.process_all_with_context_parallel(blocks)
            
            translation_time = time.time() - translation_start
            
        except Exception as e:
            logger.error(f"Ошибка при переводе: {e}")
            return False
        
        # Формируем временную метку
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
        
        # Сохраняем результаты
        logger.info("=" * 60)
        logger.info("СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
        logger.info("=" * 60)
        
        save_start = time.time()
        
        try:
            # Сохраняем только русский перевод
            output_ru = f"{input_path.stem}_{timestamp}.ru.srt"
            SubtitleProcessor.save_to_file(translated_blocks, output_ru)
            
            save_time = time.time() - save_start
            logger.info(f"  ✓ {output_ru} (русский перевод)")
                
        except Exception as e:
            logger.error(f"Ошибка при сохранении файлов: {e}")
            return False
        
        # Итоговая статистика
        total_time = time.time() - self.start_time
        
        logger.info("=" * 60)
        logger.info("ИТОГОВАЯ СТАТИСТИКА")
        logger.info("=" * 60)
        logger.info(f"✓ Всего блоков: {len(blocks)}")
        logger.info(f"✓ Переведено блоков: {self.pipeline.stats['translated']}")
        logger.info(f"✓ Язык не определен: {self.pipeline.stats['unknown_lang']}")
        logger.info(f"✓ Ошибки: {self.pipeline.stats['errors']}")
        logger.info(f"✓ Время парсинга: {self._format_time(0)}")  # Не замеряли отдельно
        logger.info(f"✓ Время перевода: {self._format_time(translation_time)}")
        logger.info(f"✓ Время сохранения: {self._format_time(save_time)}")
        logger.info(f"✓ Общее время: {self._format_time(total_time)}")
        logger.info("=" * 60)
        logger.info("✓ ГОТОВО!")
        
        return True