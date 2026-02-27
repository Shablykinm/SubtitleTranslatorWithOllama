"""Модули для определения языка и перевода текста"""
import logging
import requests
import time
from typing import Tuple, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from datetime import datetime

from subtitle_translator.models import SubtitleBlock

logger = logging.getLogger(__name__)

# Thread-local storage для безопасного логирования в потоках
thread_local = threading.local()


class ProgressBar:
    """Класс для отображения прогресс-бара с оценкой времени"""
    
    def __init__(self, total: int, description: str = "Прогресс", width: int = 50):
        self.total = total
        self.description = description
        self.width = width
        self.start_time = time.time()
        self.current = 0
        self._lock = threading.Lock()
    
    def update(self, n: int = 1):
        """Обновляет прогресс на n единиц"""
        with self._lock:
            self.current += n
            self._display()
    
    def _display(self):
        """Отображает прогресс-бар"""
        elapsed = time.time() - self.start_time
        percent = self.current / self.total if self.total > 0 else 0
        filled = int(self.width * percent)
        bar = '█' * filled + '░' * (self.width - filled)
        
        # Оценка оставшегося времени
        if self.current > 0:
            eta = (elapsed / self.current) * (self.total - self.current)
            eta_str = self._format_time(eta)
        else:
            eta_str = "?"
        
        elapsed_str = self._format_time(elapsed)
        
        print(f"\r{self.description}: |{bar}| {self.current}/{self.total} "
              f"({percent:.1%}) [Прошло: {elapsed_str}, Осталось: {eta_str}]", 
              end='', flush=True)
    
    def _format_time(self, seconds: float) -> str:
        """Форматирует время в ЧЧ:ММ:СС"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        else:
            return f"{minutes:02d}:{seconds:02d}"
    
    def finish(self):
        """Завершает прогресс-бар"""
        elapsed = time.time() - self.start_time
        elapsed_str = self._format_time(elapsed)
        print(f"\r{self.description}: |{'█' * self.width}| {self.total}/{self.total} "
              f"(100%) [Завершено за {elapsed_str}]")
        print()


class LanguageDetector:
    """Определение языка текста"""

    def __init__(self, model_manager):
        self.model_manager = model_manager
        self._lock = threading.Lock()  # Для потокобезопасного доступа к fasttext
        self.stats = {
            'detections': 0,
            'unknown': 0,
            'errors': 0
        }
        self._stats_lock = threading.Lock()

    def detect(self, text: str) -> str:
        """Возвращает код языка или 'unknown'"""
        text = text.strip()
        if not text or all(ch in '.,!?;:…- ' for ch in text):
            with self._stats_lock:
                self.stats['unknown'] += 1
                self.stats['detections'] += 1
            return "unknown"

        try:
            text_for_detection = text[:500].replace('\n', ' ')
            # Fasttext не потокобезопасен, используем блокировку
            with self._lock:
                pred = self.model_manager.lang_detector.predict(text_for_detection, k=1)
            lang = pred[0][0].replace('__label__', '')
            confidence = pred[1][0]
            
            with self._stats_lock:
                self.stats['detections'] += 1
                if confidence < 0.5:
                    self.stats['unknown'] += 1
                    
            if confidence < 0.5:
                return "unknown"
            return lang
        except Exception:
            with self._stats_lock:
                self.stats['errors'] += 1
                self.stats['detections'] += 1
            return "unknown"


class Translator:
    """Перевод текста на русский язык с помощью Ollama API"""

    def __init__(self, model_manager, config):
        self.model_manager = model_manager
        self.config = config
        self.system_prompt = config.llm_system_prompt
        self._session = None
        self.stats = {
            'translations': 0,
            'skipped_ru': 0,
            'errors': 0,
            'total_time': 0
        }
        self._stats_lock = threading.Lock()

    def _get_session(self):
        """Получить или создать сессию для текущего потока"""
        if not hasattr(thread_local, 'session'):
            thread_local.session = requests.Session()
        return thread_local.session

    def translate_to_russian(self, text: str, src_lang_code: str) -> str:
        """
        Переводит текст с исходного языка на русский.
        Использует Ollama API (локальный).
        """
        translate_start = time.time()
        
        if not text.strip():
            return text

        # Если текст уже на русском, возвращаем как есть
        if src_lang_code == "ru":
            with self._stats_lock:
                self.stats['skipped_ru'] += 1
            return text

        # Проверяем доступность Ollama
        if not self.model_manager.ollama_available:
            logger.error("Ollama не доступна, перевод невозможен.")
            with self._stats_lock:
                self.stats['errors'] += 1
            return text

        # Формируем промпт в формате Llama 3 Instruct
        if src_lang_code != "unknown":
            lang_info = f"с {src_lang_code}"
        else:
            lang_info = ""

        user_content = (
            f"Переведи следующий текст {lang_info} на русский язык, "
            f"используя правильные значения для спортивного сленга регби:\n{text}"
        )

        prompt = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{self.system_prompt}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"{user_content}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>"
        )

        # Подготовка запроса к Ollama
        url = f"{self.config.ollama_url}/api/generate"
        payload = {
            "model": self.config.ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_p": 0.95,
                "num_predict": 512,
                "stop": ["<|eot_id|>", "<|start_header_id|>"]
            }
        }

        try:
            session = self._get_session()
            response = session.post(
                url,
                json=payload,
                timeout=self.config.ollama_timeout
            )
            response.raise_for_status()
            result = response.json()
            answer = result.get("response", "").strip()
            
            with self._stats_lock:
                self.stats['translations'] += 1
                self.stats['total_time'] += time.time() - translate_start
                
            return answer
        except Exception as e:
            with self._stats_lock:
                self.stats['errors'] += 1
            logger.error(f"Ошибка перевода через Ollama: {e}")
            return text


class TranslationPipeline:
    """Конвейер перевода субтитров на русский язык с поддержкой многопоточности"""

    def __init__(self, model_manager, config):
        self.model_manager = model_manager
        self.config = config
        self.lang_detector = LanguageDetector(model_manager)
        self.translator = Translator(model_manager, config)
        self.stats = {
            'total': 0,
            'translated': 0,
            'unknown_lang': 0,
            'errors': 0
        }
        self._stats_lock = threading.Lock()
        self._progress_bar: Optional[ProgressBar] = None

    def _update_stats(self, **kwargs):
        """Потокобезопасное обновление статистики"""
        with self._stats_lock:
            for key, value in kwargs.items():
                if key in self.stats:
                    self.stats[key] += value

    def process_block(self, block: SubtitleBlock) -> SubtitleBlock:
        """
        Обрабатывает один блок - переводит на русский язык
        
        Args:
            block: исходный блок субтитров
            
        Returns:
            SubtitleBlock: переведенный блок
        """
        self._update_stats(total=1)

        try:
            if not block.text.strip():
                return SubtitleBlock(block.number, block.timestamp, block.text)

            src_lang = self.lang_detector.detect(block.text)

            if src_lang == "unknown":
                self._update_stats(unknown_lang=1)
                return SubtitleBlock(block.number, block.timestamp, block.text)

            # Переводим только на русский
            ru_text = self.translator.translate_to_russian(block.text, src_lang)

            if ru_text != block.text:
                self._update_stats(translated=1)

            return SubtitleBlock(block.number, block.timestamp, ru_text)

        except Exception as e:
            self._update_stats(errors=1)
            logger.error(f"Критическая ошибка при обработке блока {block.number}: {e}")
            return SubtitleBlock(block.number, block.timestamp, block.text)

    def process_all_parallel(self, blocks: list) -> list:
        """
        Обрабатывает все блоки параллельно с использованием ThreadPoolExecutor
        
        Args:
            blocks: список блоков субтитров
            
        Returns:
            list: список переведенных блоков
        """
        translated_blocks = [None] * len(blocks)
        total = len(blocks)

        logger.info(f"▶ Начинаем параллельный перевод на русский язык ({self.config.max_workers} потоков)")
        logger.info(f"▶ Всего блоков для обработки: {total}")
        logger.info("-" * 60)

        self.stats = {k: 0 for k in self.stats}
        self._progress_bar = ProgressBar(total, "Перевод блоков")

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Создаем словарь будущих результатов
            future_to_index = {
                executor.submit(self.process_block, block): i 
                for i, block in enumerate(blocks)
            }

            completed = 0
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    translated_block = future.result(timeout=30)
                    translated_blocks[index] = translated_block
                except Exception as e:
                    logger.error(f"Ошибка в потоке для блока {index}: {e}")
                    translated_blocks[index] = blocks[index]
                    self._update_stats(errors=1)

                completed += 1
                self._progress_bar.update()

        self._progress_bar.finish()
        
        # Детальная статистика
        logger.info("-" * 60)
        logger.info("СТАТИСТИКА ПЕРЕВОДА:")
        logger.info(f"  ✓ Всего блоков: {total}")
        logger.info(f"  ✓ Переведено: {self.stats['translated']}")
        logger.info(f"  ⚠ Язык не определен: {self.stats['unknown_lang']}")
        logger.info(f"  ✗ Ошибки: {self.stats['errors']}")
        
        # Статистика от детектора языка
        logger.info("  📊 Определение языка:")
        logger.info(f"     - Всего определений: {self.lang_detector.stats['detections']}")
        logger.info(f"     - Неопределено: {self.lang_detector.stats['unknown']}")
        logger.info(f"     - Ошибки: {self.lang_detector.stats['errors']}")
        
        # Статистика от переводчика
        if self.translator.stats['translations'] > 0:
            avg_time = self.translator.stats['total_time'] / self.translator.stats['translations']
            logger.info("  📊 Перевод:")
            logger.info(f"     - Выполнено переводов: {self.translator.stats['translations']}")
            logger.info(f"     - Пропущено (русский): {self.translator.stats['skipped_ru']}")
            logger.info(f"     - Ошибки API: {self.translator.stats['errors']}")
            logger.info(f"     - Среднее время перевода: {avg_time:.2f} сек")

        return translated_blocks

    def _adjust_split_point(self, translated_text: str, desired_end: int, search_range: int = 50) -> int:
        """Корректирует позицию окончания блока"""
        text_len = len(translated_text)
        if desired_end <= 0 or desired_end >= text_len:
            return desired_end

        left_bound = max(0, desired_end - search_range)
        right_bound = min(text_len, desired_end + search_range)

        best_pos = desired_end
        min_dist = search_range + 1

        for i in range(left_bound, right_bound):
            if translated_text[i] in '.!?':
                end_candidate = i + 1
                if end_candidate < text_len and translated_text[end_candidate] == ' ':
                    end_candidate += 1
                dist = abs(end_candidate - desired_end)
                if dist < min_dist:
                    min_dist = dist
                    best_pos = end_candidate

        if min_dist <= search_range:
            return best_pos

        for i in range(left_bound, right_bound):
            if translated_text[i] == ' ':
                end_candidate = i + 1
                dist = abs(end_candidate - desired_end)
                if dist < min_dist:
                    min_dist = dist
                    best_pos = end_candidate

        return best_pos

    def _split_translated_text(self, original_blocks: List[str], translated_text: str) -> List[str]:
        """
        Разделяет переведенный текст на блоки с учетом целостности слов
        и предотвращает появление пустых блоков с одиночными буквами.
        
        Args:
            original_blocks: список оригинальных текстов блоков
            translated_text: переведенный текст группы
            
        Returns:
            List[str]: список текстов для каждого блока
        """
        # Если группа состоит из одного блока, возвращаем весь текст
        if len(original_blocks) == 1:
            return [translated_text]
        
        # Если переведенный текст пустой, возвращаем пустые строки
        if not translated_text.strip():
            return [''] * len(original_blocks)
        
        # Разбиваем переведенный текст на слова
        words = translated_text.split()
        
        # Если слов меньше, чем блоков, распределяем слова по блокам,
        # но не оставляем блоки пустыми
        if len(words) < len(original_blocks):
            result = []
            for i in range(len(original_blocks)):
                if i < len(words):
                    result.append(words[i])
                else:
                    result.append('')
            return result
        
        # Рассчитываем пропорции на основе длины оригинальных блоков
        total_original_len = sum(len(block) for block in original_blocks)
        if total_original_len == 0:
            return [''] * len(original_blocks)
        
        # Распределяем слова по блокам пропорционально
        result = []
        word_index = 0
        
        for i, original_block in enumerate(original_blocks):
            # Пропорция для текущего блока
            block_ratio = len(original_block) / total_original_len
            words_for_block = max(1, int(len(words) * block_ratio))
            
            # Для последнего блока берем все оставшиеся слова
            if i == len(original_blocks) - 1:
                block_words = words[word_index:]
            else:
                block_words = words[word_index:word_index + words_for_block]
                word_index += words_for_block
            
            # Собираем текст блока
            if block_words:
                block_text = ' '.join(block_words)
            else:
                block_text = ''
            
            result.append(block_text)
        
        # Пост-обработка: если какой-то блок пустой, объединяем его с соседним
        i = 0
        while i < len(result):
            if not result[i].strip() and len(result) > 1:
                # Пустой блок - объединяем с предыдущим или следующим
                if i > 0:
                    # Объединяем с предыдущим
                    result[i-1] = result[i-1] + ' ' + result[i]
                    result.pop(i)
                    continue
                elif i < len(result) - 1:
                    # Объединяем со следующим
                    result[i] = result[i] + ' ' + result[i+1]
                    result.pop(i+1)
                    continue
            i += 1
        
        # Убеждаемся, что количество блоков соответствует оригиналу
        while len(result) < len(original_blocks):
            result.append('')
        while len(result) > len(original_blocks):
            # Объединяем лишние блоки с последним
            result[-2] = result[-2] + ' ' + result[-1]
            result.pop()
        
        return result

    def _is_end_of_sentence(self, text: str) -> bool:
        return text.strip().endswith(('.', '!', '?'))

    def _is_short_block(self, text: str) -> bool:
        """Проверяет, является ли блок слишком коротким (1-2 слова)"""
        stripped = text.strip()
        if not stripped:
            return False
        words = stripped.split()
        return len(words) <= 2 and len(stripped) < 20

    def process_all_with_context_parallel(self, blocks: list, context_size: int = None) -> list:
        """
        Обрабатывает блоки с контекстом параллельно, переводит на русский язык
        
        Args:
            blocks: список блоков субтитров
            context_size: размер контекста (максимальное количество блоков в группе)
            
        Returns:
            list: список переведенных блоков
        """
        if context_size is None:
            context_size = self.config.translation_batch_size

        translated_blocks = [None] * len(blocks)
        total = len(blocks)

        logger.info(f"▶ Начинаем контекстный перевод на русский язык")
        logger.info(f"▶ Параметры: {self.config.max_workers} потоков, макс. {context_size} блоков в группе")
        logger.info(f"▶ Всего блоков: {total}")
        logger.info("-" * 60)

        # Формируем группы блоков с учетом коротких блоков
        group_start_time = time.time()
        groups = []
        group_indices = []  # список кортежей (start_index, end_index)
        
        i = 0
        while i < total:
            group_start = i
            group_end = i
            short_blocks_in_group = 0
            
            while group_end < total and (group_end - group_start) < context_size:
                current_text = blocks[group_end].text.strip()
                
                # Если блок пустой, пропускаем его (будет обработан отдельно)
                if not current_text:
                    group_end += 1
                    continue
                
                # Проверяем, является ли блок коротким
                if self._is_short_block(current_text):
                    short_blocks_in_group += 1
                
                # Если накопилось слишком много коротких блоков, заканчиваем группу
                if short_blocks_in_group >= 3 and group_end > group_start:
                    break
                
                # Проверяем конец предложения
                if self._is_end_of_sentence(current_text) and group_end != group_start:
                    group_end += 1
                    break
                    
                group_end += 1
            
            # Убеждаемся, что группа не пустая
            if group_end == group_start:
                group_end = group_start + 1
            
            groups.append(blocks[group_start:group_end])
            group_indices.append((group_start, group_end))
            i = group_end

        group_formation_time = time.time() - group_start_time
        logger.info(f"✓ Сформировано {len(groups)} групп за {group_formation_time:.2f} сек")
        logger.info(f"  Средний размер группы: {total/len(groups):.1f} блоков")
        logger.info("-" * 60)

        # Функция для обработки одной группы
        def process_group(group_data):
            group_blocks, start_idx, end_idx = group_data
            original_texts = [b.text for b in group_blocks]
            combined_text = " ".join(original_texts)

            # Определяем язык группы
            src_lang = "unknown"
            for b in group_blocks:
                if b.text.strip():
                    src_lang = self.lang_detector.detect(b.text)
                    if src_lang != "unknown":
                        break
            if src_lang == "unknown":
                src_lang = self.lang_detector.detect(combined_text)

            if src_lang == "unknown":
                # Язык не определен - возвращаем оригиналы
                return {
                    'start': start_idx,
                    'end': end_idx,
                    'blocks': group_blocks,
                    'translated': False
                }

            # Если текст уже на русском, возвращаем как есть
            if src_lang == "ru":
                return {
                    'start': start_idx,
                    'end': end_idx,
                    'blocks': group_blocks,
                    'translated': False
                }

            # Переводим группу на русский
            ru_combined = self.translator.translate_to_russian(combined_text, src_lang)

            # Разделяем обратно на блоки с учетом целостности слов
            ru_split = self._split_translated_text(original_texts, ru_combined)

            # Создаем блоки с сохранением номеров и временных меток
            ru_result = []
            for j, block in enumerate(group_blocks):
                ru_result.append(SubtitleBlock(block.number, block.timestamp, ru_split[j]))

            return {
                'start': start_idx,
                'end': end_idx,
                'blocks': ru_result,
                'translated': True
            }

        # Подготавливаем данные для параллельной обработки
        group_data = [(groups[j], group_indices[j][0], group_indices[j][1]) for j in range(len(groups))]
        
        # Прогресс-бар для групп
        group_progress = ProgressBar(len(groups), "Обработка групп")

        # Запускаем параллельную обработку групп
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = [executor.submit(process_group, data) for data in group_data]

            completed = 0
            translated_count = 0
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=60)
                    start, end = result['start'], result['end']
                    
                    # Вставляем результаты в правильные позиции
                    for j in range(end - start):
                        translated_blocks[start + j] = result['blocks'][j]
                    
                    if result['translated']:
                        translated_count += 1
                except Exception as e:
                    logger.error(f"Ошибка при обработке группы: {e}")

                completed += 1
                group_progress.update()

        group_progress.finish()
        
        # Детальная статистика
        logger.info("-" * 60)
        logger.info("СТАТИСТИКА КОНТЕКСТНОГО ПЕРЕВОДА:")
        logger.info(f"  ✓ Всего блоков: {total}")
        logger.info(f"  ✓ Всего групп: {len(groups)}")
        logger.info(f"  ✓ Переведено групп: {translated_count}")
        logger.info(f"  ⚠ Групп с неопределенным языком: {len(groups) - translated_count}")
        
        # Статистика от детектора языка
        logger.info("  📊 Определение языка:")
        logger.info(f"     - Всего определений: {self.lang_detector.stats['detections']}")
        logger.info(f"     - Неопределено: {self.lang_detector.stats['unknown']}")
        logger.info(f"     - Ошибки: {self.lang_detector.stats['errors']}")
        
        # Статистика от переводчика
        if self.translator.stats['translations'] > 0:
            avg_time = self.translator.stats['total_time'] / self.translator.stats['translations']
            logger.info("  📊 Перевод:")
            logger.info(f"     - Выполнено переводов групп: {self.translator.stats['translations']}")
            logger.info(f"     - Пропущено (русский): {self.translator.stats['skipped_ru']}")
            logger.info(f"     - Ошибки API: {self.translator.stats['errors']}")
            logger.info(f"     - Среднее время перевода группы: {avg_time:.2f} сек")

        return translated_blocks