## 🤖 Построение RAG-модели с нуля - умный советник онлайн-курсов

*Данные были взяты из открытых источников karpov.courses на главной странице*

<p align="center">
  <img src="https://github.com/user-attachments/assets/5007ded4-e048-47dd-aa5d-01b0cb4053c2"/>
</p>

## Описание проекта 🚀

* Построенный на основе RAG умный чат-бот на основе технологий Mistral AI + LangChain с визуализацией в Streamlit.
* Все данные парсятся с помощью BeautifulSoup4 и requests полностью автоматически.
* С помощью zero-shot-классификации моделью 🤗 [DeepPavlov/rubert-base-cased-sentence](https://huggingface.co/DeepPavlov/rubert-base-cased-sentence) спарсенные данные классифицируются для лучшей структуры.
* Данные после классификации векторизируются с помощью модели 🤗 [cointegrated/rubert-base-cased-nli-threeway](https://huggingface.co/cointegrated/rubert-base-cased-nli-threeway)
* Спарсенные данные надёжно хранятся в векторной БД Qdrant.
* Чат-бот на основе RAG можно переделать под любые другие данные, нужно лишь изменить код парсинга под свои нужды.

## Использованные технологии ⚙️
<p align="center">
  <a href="https://go-skill-icons.vercel.app/">
    <img src="https://go-skill-icons.vercel.app/api/icons?i=linux,python,pycharm,langchain,numpy,docker,streamlit,pandas,qdrant,huggingface,mistral&theme=dark"/>
  </a>
</p>

## ⚙️ Структура репозитория **rag-courses-advicer**

```
rag-courses-advicer/                # Основная папка проекта
├── LICENSE                         # Лицензия проекта
├── notebook
│   └── rag-chatbot-pipeline-full.ipynb  # Ноутбук с полным решением задания
├── requirements.txt                # Зависимости проекта
└── streamlit_app                   # Демонстрация через Streamlit
```

### 📁 Основные директории:
- **`LICENSE`**: Лицензия проекта (Apache 2.0).
- **`Makefile`**: Скрипт для сборки проекта (установка, выполнение тестов заданий).
- **`rag-chatbot-pipeline-full.ipynb`**: Ноутбук с полным решением задания от парсинга до построения конечного чат-бота (без визуализации в Streamlit).
- **`requirements.txt`**: Список зависимостей проекта.
- **`streamlit_app/`**: Демонстрация работы чат-бота в виде веб-приложения в Streamlit.

---

### ⚠️ Важное замечание:

* Для полноценной работы потребуется API от Mistral AI.
