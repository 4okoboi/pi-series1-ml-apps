from transformers import pipeline
from typing import Optional


def main(
        pipeline_name: str,
        text_for_classification: str,
        model_name: Optional[str] = None
):
    """
    Функция для вызова Pipeline. По умолчанию pipeline использует модель `distilbert/distilbert-base-uncased-finetuned-sst-2-english`
    :param text_for_classification: текст, с которым будем инференсить модель
    :param pipeline_name: название Pipeline (из разрешенных от HuggingFace)
    :param model_name: название модели, если нужно использовать не деефолтную
    :return:
    """
    # Объявляем классификатор
    classifier = pipeline(task=pipeline_name, model=model_name)

    # инференсим классификатор с текстом, который пришел на вход функции
    output = classifier(text_for_classification)
    print(
        f"Текст: {text_for_classification} модель определила как: {output[0]['label']} с вреоятностью: {output[0]['score']}")

# На русском
main(pipeline_name="sentiment-analysis",
     text_for_classification="Привет! Тот фильм был очень крут. Спасибо тебе!",
     model_name="blanchefort/rubert-base-cased-sentiment"
)

# На иностранском
main(
    pipeline_name="sentiment-analysis",
    text_for_classification="Hi! This was Great! Thank you"
)