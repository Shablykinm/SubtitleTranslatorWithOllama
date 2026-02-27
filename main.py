#!/usr/bin/env python3
"""
Использование: python main.py <входной_файл.srt>
"""
import argparse
import logging
import sys
import os

# Добавляем текущую директорию в PATH для корректного импорта
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from subtitle_translator import SubtitleTranslatorApp

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)


def main():
    """Точка входа"""
    parser = argparse.ArgumentParser(
        description="Переводчик субтитров",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python main.py audio.srt
  python main.py subtitles.txt
        """
    )
    parser.add_argument(
        'input_file',
        help='Входной файл субтитров (.srt или .txt)'
    )
    
    args = parser.parse_args()
    
    # Создаем и запускаем приложение
    app = SubtitleTranslatorApp()
    
    if not app.initialize():
        sys.exit(1)
    
    if not app.run(args.input_file):
        sys.exit(1)


if __name__ == "__main__":
    main()