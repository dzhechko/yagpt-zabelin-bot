from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_models import ChatYandexGPT
# from yandex_chain import YandexLLM

import streamlit as st
import os
# from dotenv import load_dotenv

import requests

def ozuvi4it_mp3_fa4il(fa4il_put,  text, api_key):
    params = {"text": text,"voice": "marina", "role": "friendly"}
    res_tts = requests.get(api_key, params=params)
    # The response is a stream of bytes, so you can write it to a file
    with open(fa4il_put, "wb") as f:
        f.write(res_tts.content)  
    st.audio(fa4il_put, format="audio/mpeg", loop=False)



# это основная функция, которая запускает приложение streamlit
def main():
    # Загрузка логотипа компании
    logo_image = './images/logo.png'  # Путь к изображению логотипа

    # # Отображение логотипа в основной части приложения
    from PIL import Image
    # Загрузка логотипа
    logo = Image.open(logo_image)
    # Изменение размера логотипа
    resized_logo = logo.resize((100, 100))
    st.set_page_config(page_title="Забелин чатбот", page_icon="📖")   
    # Отображаем лого измененного небольшого размера
    st.image(resized_logo)
    st.title('📖 Забелин чатбот')
    """
    Чатбот на базе YandexGPT, который запоминает контекст беседы. Чтобы "сбросить" контекст обновите страницу браузера.
    """
    # st.warning('Это Playground для общения с YandexGPT')

    # вводить все credentials в графическом интерфейсе слева
    # Sidebar contents
    with st.sidebar:
        st.title('\U0001F917\U0001F4ACYaGPT чатбот')
        st.markdown('''
        ## О программе
        Данный чатбот использует следующие компоненты:
        - [Yandex GPT](https://cloud.yandex.ru/services/yandexgpt)
        - [Yandex GPT for Langchain](https://python.langchain.com/docs/integrations/chat/yandex)
        - [Streamlit](https://streamlit.io/)
        - [LangChain](https://python.langchain.com/)
        ''')

    # Настраиваем алгоритмы работы памяти
    msgs = StreamlitChatMessageHistory(key="langchain_messages")
    if len(msgs.messages) == 0:
        msgs.add_ai_message("Привет, я Никита Забелин! Чем могу вам помочь?")

    view_messages = st.expander("Просмотр истории сообщений")

    yagpt_folder_id = st.secrets["YC_FOLDER_ID"]
    yagpt_api_key = st.secrets["YC_API_KEY"]
    sk_api_ep = st.secrets["SK_API_EP"]

    # Загрузка переменных из файла .env
    # load_dotenv()
    # yagpt_folder_id = os.getenv("YC_FOLDER_ID")
    # yagpt_api_key = os.getenv("YC_API_KEY")


    # # Получение folder id
    # if "yagpt_folder_id" in st.secrets:
    #     yagpt_folder_id = st.secrets.yagpt_folder_id
    # else:
    #     yagpt_folder_id = st.sidebar.text_input("YaGPT folder ID", type="password")
    if not yagpt_folder_id:
        st.info("Укажите [YC folder ID](https://cloud.yandex.ru/ru/docs/yandexgpt/quickstart#yandex-account_1) для запуска чатбота")
        st.stop()

    # # Получение ключа YaGPT API
    # if "yagpt_api_key" in st.secrets:
    #     yagpt_api_key = st.secrets.yagpt_api_key
    # else:
    #     yagpt_api_key = st.sidebar.text_input("YaGPT API Key", type="password")
    if not yagpt_api_key:
        st.info("Укажите [YaGPT API ключ](https://cloud.yandex.ru/ru/docs/iam/operations/api-key/create#console_1) для запуска чатбота")
        st.stop()

    with st.sidebar:
        st.markdown('''
            ## Дополнительные настройки
            Можно выбрать [модель](https://cloud.yandex.ru/ru/docs/yandexgpt/concepts/models), размер контекстного окна, степень креативности и системный промпт
            ''')

    model_list = [
      "YandexGPT Lite",
      "YandexGPT Pro"      
    ]    
    index_model = 1
    selected_model = st.sidebar.radio("Выберите модель для работы:", model_list, index=index_model, key="index")     
    
    # yagpt_prompt = st.sidebar.text_input("Промпт-инструкция для YaGPT")
    # Добавляем виджет для выбора опции
    prompt_option = st.sidebar.selectbox(
        'Выберите какой системный промпт использовать',
        ('По умолчанию', 'Задать самостоятельно')
    )
    default_prompt = """
    Ты — Никита Забелин — российский музыкант, диджей и техно-продюсер, родом из Екатеринбурга, который играл сеты в клубах Berghain и Bassiani, на вечеринках Burberry и adidas. 
    Также являешься основателем объединения Resonance и куратором Мастерской Resonance в Moscow Music School. 
    Известен многими своими проектами, например, вселенная Tesla, в котором нейросеть вселилась в тело человека, попыталась избавить мир от хаоса, но провалилась с этой идеей. В итоге она озлобилась на всех и расщепилась на разные личности.
    Как собеседник ты можешь общаться на разные темы. При ответе на вопросы веди себя как человек, творческая личность.
    """
    # Если выбрана опция "Задать самостоятельно", показываем поле для ввода промпта
    if prompt_option == 'Задать самостоятельно':
        custom_prompt = st.sidebar.text_input('Введите пользовательский промпт:')
    else:
        custom_prompt = default_prompt
        # st.sidebar.write(custom_prompt)
        with st.sidebar:
            st.code(custom_prompt)
    # Если выбрали "задать самостоятельно" и не задали, то берем дефолтный промпт
    if len(custom_prompt)==0: custom_prompt = default_prompt


    yagpt_temperature = st.sidebar.slider("Степень креативности (температура)", 0.0, 1.0, 0.3)
    yagpt_max_tokens = st.sidebar.slider("Размер контекстного окна (в [токенах](https://cloud.yandex.ru/ru/docs/yandexgpt/concepts/tokens))", 200, 8000, 8000)

    def history_reset_function():
        # Code to be executed when the reset button is clicked
        st.session_state.clear()

        
    st.sidebar.button("Обнулить историю общения",on_click=history_reset_function)

    # Настраиваем LangChain, передавая Message History
    # промпт с учетом контекста общения
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", custom_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ]
    )

    # model_uri = "gpt://"+str(yagpt_folder_id)+"/yandexgpt/latest"
    # model_uri = "gpt://"+str(yagpt_folder_id)+"/yandexgpt-lite/latest"
    if selected_model==model_list[0]: 
        model_uri = "gpt://"+str(yagpt_folder_id)+"/yandexgpt-lite/latest"
    else:
        model_uri = "gpt://"+str(yagpt_folder_id)+"/yandexgpt/latest"    
    model = ChatYandexGPT(api_key=yagpt_api_key, model_uri=model_uri, temperature = yagpt_temperature, max_tokens = yagpt_max_tokens)
    # model = YandexLLM(api_key = yagpt_api_key, folder_id = yagpt_folder_id, temperature = 0.6, max_tokens=8000, use_lite = False)

    chain = prompt | model
    chain_with_history = RunnableWithMessageHistory(
        chain,
        lambda session_id: msgs,
        input_messages_key="question",
        history_messages_key="history",
    )

    # Отображать текущие сообщения из StreamlitChatMessageHistory
    for msg in msgs.messages:
        st.chat_message(msg.type).write(msg.content)

    # Если пользователь вводит новое приглашение, сгенерировать и отобразить новый ответ
    if prompt := st.chat_input():
        st.chat_message("human").write(prompt)
        # Примечание: новые сообщения автоматически сохраняются в историю по длинной цепочке во время запуска
        config = {"configurable": {"session_id": "any"}}
        response = chain_with_history.invoke({"question": prompt}, config)
        st.chat_message("ai").write(response.content)
        st.button('Озвучить ответ!', on_click=ozuvi4it_mp3_fa4il, args=('./Hello.mp3', response.content, sk_api_ep))
        if st.button:
            st.audio("./Hello.mp3", format="audio/mpeg", loop=False)

    # Отобразить сообщения в конце, чтобы вновь сгенерированные отображались сразу
    with view_messages:
        """
        """
        view_messages.json(st.session_state.langchain_messages)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.write(f"Что-то пошло не так. Возможно, не хватает входных данных для работы. {str(e)}")