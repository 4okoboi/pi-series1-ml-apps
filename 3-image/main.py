from typing import List

from transformers import pipeline
from PIL import Image


def classify_image(
        image_path: str,
        labels: List[str],
        model_name: str = "openai/clip-vit-large-patch14"
):
    """
    Фунуция для инференса модели
    :param labels: лейблы типов изображения
    :param image_path: путь к файлу
    :param model_name: название модели, которая будет использоваться в pipeline
    :return:
    """
    # Инициализация пайплайна
    classifier = pipeline(type="zero-shot-image-classification", model=model_name)

    image = Image.open(image_path)

    # инференс
    output = classifier(image=image, candidate_labels=labels)

    if isinstance(output, dict):
        res = [output]
    else:
        res = output

    # берем результат с лучшим скором
    res = sorted(res, key=lambda d: d["score"], reverse=True)[0]

    return res


if __name__ == "__main__":
    # передаем лейблы и путь к изображению
    image_path = input("Введите путь к изображению: ").strip()
    result = classify_image(
        image_path=image_path,
        labels=["house", "car", "document"]
    )
    print(f"Фото: {image_path} имеет класс {result.get('label')} с вероятностью {result.get('score')}")
