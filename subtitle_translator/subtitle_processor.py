"""Обработка SRT файлов"""
import re
from typing import List

from subtitle_translator.models import SubtitleBlock


class SubtitleProcessor:
    """Обработка SRT файлов"""
    
    @staticmethod
    def parse_file(filepath: str) -> List[SubtitleBlock]:
        """
        Парсит SRT файл и возвращает список блоков субтитров
        
        Args:
            filepath: путь к файлу субтитров
            
        Returns:
            List[SubtitleBlock]: список блоков субтитров
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Нормализуем переносы строк - заменяем все виды переносов на \n
        content = content.replace('\r\n', '\n').replace('\r', '\n')
        
        # Улучшенное регулярное выражение для блоков SRT
        # Более гибкое - учитывает возможные пробелы и разные форматы
        pattern = re.compile(
            r'(\d+)\s*\n(\d{2}:\d{2}:\d{2}[,.]\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}[,.]\d{3})\s*\n(.*?)(?=\n\s*\d+\s*\n|\n*$|\Z)',
            re.DOTALL | re.MULTILINE
        )
        
        blocks = []
        for match in pattern.finditer(content):
            # Очищаем текст от лишних пробелов и переносов
            text = match.group(3).strip()
            # Заменяем множественные переносы строк на пробелы
            text = re.sub(r'\s+', ' ', text)
            
            block = SubtitleBlock(
                number=match.group(1).strip(),
                timestamp=match.group(2).strip(),
                text=text
            )
            blocks.append(block)
        
        # Если не нашли блоки стандартным regex, пробуем альтернативный метод
        if not blocks:
            blocks = SubtitleProcessor._parse_alternative(content)
        
        return blocks
    
    @staticmethod
    def _parse_alternative(content: str) -> List[SubtitleBlock]:
        """
        Альтернативный метод парсинга для сложных случаев
        
        Args:
            content: содержимое файла
            
        Returns:
            List[SubtitleBlock]: список блоков субтитров
        """
        blocks = []
        lines = content.replace('\r\n', '\n').replace('\r', '\n').split('\n')
        
        i = 0
        while i < len(lines):
            # Пропускаем пустые строки
            if not lines[i].strip():
                i += 1
                continue
            
            # Ищем номер блока
            if lines[i].strip().isdigit():
                number = lines[i].strip()
                i += 1
                
                # Ищем временную метку
                if i < len(lines) and '-->' in lines[i]:
                    timestamp = lines[i].strip()
                    i += 1
                    
                    # Собираем текст блока
                    text_lines = []
                    while i < len(lines) and lines[i].strip():
                        text_lines.append(lines[i].strip())
                        i += 1
                    
                    text = ' '.join(text_lines)
                    
                    if text:  # Добавляем только непустые блоки
                        blocks.append(SubtitleBlock(
                            number=number,
                            timestamp=timestamp,
                            text=text
                        ))
                else:
                    i += 1
            else:
                i += 1
        
        return blocks
    
    @staticmethod
    def save_to_file(blocks: List[SubtitleBlock], filename: str):
        """
        Сохраняет блоки субтитров в файл
        
        Args:
            blocks: список блоков субтитров
            filename: имя выходного файла
        """
        with open(filename, 'w', encoding='utf-8', newline='\n') as f:
            for block in blocks:
                f.write(f"{block.number}\n{block.timestamp}\n{block.text}\n\n")