## YandexGPT чатбот с поддержкой контекста общения с пользователем 

### Краткая информация
Данный YandexGPT-бот использует следующие компоненты:
- [Yandex GPT](https://cloud.yandex.ru/services/yandexgpt)
- [Yandex GPT for Langchain](https://python.langchain.com/docs/integrations/chat/yandex)
- [Streamlit](https://streamlit.io/)
- [LangChain](https://python.langchain.com/)

### Структура репозитория и порядок работы с ботом
- в файле ``.env`` находятся системные переменные (которые при запуске в облаке можно указать как secrets)
```
YAGPT_FOLDER_ID = 
YAGPT_API_KEY = 
```
- файл `requirements.txt` традиционно содержит в себе список необходимых для работы программы модулей, которые устанавливаются командой 
```pip install -r requirements.txt ```
- в папке `images` хранится логотип компании, который можно использовать в графическом интерфейсе streamlit
- `yagpt-chat-with-history.py` запускаемый файл. Адаптирован для запуска в [Streamlit Community Cloud](https://docs.streamlit.io/streamlit-community-cloud/get-started). Требует задания системных переменных в streamlit secrets при разворачивании приложения.

### Запуск в Streamlit Community Cloud
Вы можете развернуть данное приложение через Streamlit Community Cloud, следуя [инструкциям](https://docs.streamlit.io/streamlit-community-cloud/get-started)

