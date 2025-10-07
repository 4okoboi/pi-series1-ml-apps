import argparse
import json
import os
from typing import Any, Dict, List
from unittest import result

import torch
import whisper


def parse_args():
    parser = argparse.ArgumentParser(description='Pytorch simple audio transcription')
    parser.add_argument('--input', required=True, help='path to audio file')
    parser.add_argument('--output', default='.', help='path to output folder')
    parser.add_argument('--language', required=True, choices=["en", "ru"], help='audio language')
    parser.add_argument("--txt", action="store_true", help="save TXT")
    parser.add_argument("--json", action="store_true", help="save JSON")
    return parser.parse_args()


def transcribe_file(
        input_path: str,
        language: str,
) -> Dict[str, Any]:
    """
    Функция для траскрибации аудиофайла. Загружает модель, собирает настройки и инференсит.
    :param input_path: путь к аудио
    :param language: язык речи
    :return:
    """
    # выбор девайса, на которм будет инференситься
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # загрузка модели
    model = whisper.load_model('small', device=device)

    # инизиацлизация параметров
    options = {
        "task": "transcribe",
        "language": language,
        "fp16": device == 'cuda',
        "condition_on_previous_text": True
    }

    # инференс
    result = model.transcribe(
        audio=input_path,
        **options
    )
    return result


def save_output(
        model_result: Dict[str, Any],
        input_filename: str,
        output_folder: str,
        is_json: bool,
        is_txt: bool
) -> List[str]:
    """
    Функция для сохранения транскрибации в требуемом формате
    :param model_result:
    :param input_filename:
    :param output_folder:
    :param is_json:
    :param is_txt:
    :return:
    """
    result_paths = []
    if not is_json and not is_txt:
        return None

    if is_json:
        with open(os.path.join(output_folder, input_filename + '.json'), 'w', encoding='utf-8') as f:
            json.dump({
                "text": model_result.get('text', ''),
                "segments": model_result.get('segments', []),
            }, f, ensure_ascii=False, indent=4)
            result_paths.append(os.path.join(output_folder, input_filename + '.json'))
    if is_txt:
        with open(os.path.join(output_folder, input_filename + '.txt'), 'w', encoding='utf-8') as f:
            f.write(model_result.get('text', ''))
            result_paths.append(os.path.join(output_folder, input_filename + '.txt'))

    return result_paths


def main():
    args = parse_args()
    try:
        input_file_name = os.path.basename(args.input).split('.')[0]
    except Exception:
        input_file_name = os.path.basename(args.input)
    # получаем транскрибацию
    transcribe_result = transcribe_file(args.input, args.language)

    # сохраняем транскрибацию в нужном формате (txt, json)
    output_paths = save_output(transcribe_result, input_file_name, args.output, args.json, args.txt)
    if not output_paths:
        print(f"Транскрибированный текст: {transcribe_result.get('text')}")
    else:
        print(f"Сохранено в: {output_paths}")


if __name__ == "__main__":
    main()
