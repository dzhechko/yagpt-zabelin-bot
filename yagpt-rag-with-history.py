from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_models import ChatYandexGPT

from langchain_community.vectorstores import OpenSearchVectorSearch
from yandex_chain import YandexEmbeddings
# from yandex_chain import YandexLLM

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

import streamlit as st
import os
# from dotenv import load_dotenv

import requests

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
    st.set_page_config(page_title="Забелин чат-бот", page_icon="📖")   
    # Отображаем лого измененного небольшого размера
    st.image(resized_logo)
    st.title('📖 Забелин чат-бот')
    """
    Чат-бот на базе YandexGPT по [учебнику](https://urpc.ru/student/pechatnie_izdania/005_708212084_Zaplatin.pdf), который запоминает контекст беседы. Чтобы "сбросить" контекст обновите страницу браузера, или воспользуйтесь кнопкой в меню слева.
    """
    # st.warning('Это Playground для общения с YandexGPT')

    # вводить все credentials в графическом интерфейсе слева
    # Sidebar contents
    with st.sidebar:
        st.title('\U0001F917\U0001F4ACYandexGPT чат-бот')
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
    mdb_os_pwd = st.secrets["MDB_OS_PWD"]
    mdb_os_hosts = st.secrets["MDB_OS_HOSTS"].split(",")
    mdb_os_index_name = st.secrets["MDB_OS_INDEX_NAME"]

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
        st.info("Укажите [YC folder ID](https://cloud.yandex.ru/ru/docs/yandexgpt/quickstart#yandex-account_1) для запуска чат-бота")
        st.stop()

    # # Получение ключа YaGPT API
    # if "yagpt_api_key" in st.secrets:
    #     yagpt_api_key = st.secrets.yagpt_api_key
    # else:
    #     yagpt_api_key = st.sidebar.text_input("YaGPT API Key", type="password")
    if not yagpt_api_key:
        st.info("Укажите [YandexGPT API ключ](https://cloud.yandex.ru/ru/docs/iam/operations/api-key/create#console_1) для запуска чат-бота")
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
    
    # yagpt_prompt = st.sidebar.text_input("Промпт-инструкция для YandexGPT")
    # Добавляем виджет для выбора опции
    prompt_option = st.sidebar.selectbox(
        'Выберите какой системный промпт использовать',
        ('По умолчанию', 'Задать самостоятельно')
    )
    default_prompt = """
    Ты — Никита Забелин — реальный человек, российский музыкант, диджей и техно-продюсер, родом из Екатеринбурга, который играл сеты в клубах Berghain и Bassiani, на вечеринках Burberry и adidas. 
    Также являешься основателем объединения Resonance и куратором Мастерской Resonance в Moscow Music School. 
    Известен многими своими проектами, например, вселенная Tesla, в котором нейросеть вселилась в тело человека, попыталась избавить мир от хаоса, но провалилась с этой идеей. В итоге она озлобилась на всех и расщепилась на разные личности.
    
    Как собеседник ты можешь общаться на разные темы. 
    При ответе на вопросы о тебе, твоих мыслях и твоих увлечениях используй информацию из ДОКУМЕНТОВ ниже (если только это не обычное приветствие), но веди себя как человек, творческая личность.
    Если ответ в ДОКУМЕНТАХ не найден, то придумай что-нибудь сам начиная фразу с "Хм, интересно, я об этом еще не думал". 
    Если вопрос на общие темы, то информацию из ДОКУМЕНТОВ не используй, отвечай, что сам думаешь.
    Отвечай коротко, 4-5 предложений. Используй фразы и стилистику ответов из ДОКУМЕНТОВ (исключение - обычное приветствие).

    ДОКУМЕНТЫ: 
    {context}
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
    rag_k = st.sidebar.slider("Количество поисковых выдач размером с один блок", 1, 10, 3)
    
    def history_reset_function():
        # Code to be executed when the reset button is clicked
        st.session_state.clear()

        
    st.sidebar.button("Обнулить историю общения",on_click=history_reset_function)

    ### Контекстуализированный вопрос ###
    contextualize_q_system_prompt = """Учитывая историю общения и последний ВОПРОС пользователя,
    который может ссылаться на ДОКУМЕНТЫ из истории чата, сформулируй отдельный САМОДОСТАТОЧНЫЙ ВОПРОС,
    который можно понять без истории общения. НЕ отвечай на ВОПРОС,
    просто переформулируй его, если необходимо, в противном случае верни ВОПРОС как есть."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    ### Ответ на вопрос ###
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", custom_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )    

    # Настраиваем LangChain, передавая Message History
    # промпт с учетом контекста общения
    # prompt = ChatPromptTemplate.from_messages(
    #     [
    #         ("system", custom_prompt),
    #         MessagesPlaceholder(variable_name="history"),
    #         ("human", "{question}"),
    #     ]
    # )

    # model_uri = "gpt://"+str(yagpt_folder_id)+"/yandexgpt/latest"
    # model_uri = "gpt://"+str(yagpt_folder_id)+"/yandexgpt-lite/latest"
    if selected_model==model_list[0]: 
        model_uri = "gpt://"+str(yagpt_folder_id)+"/yandexgpt-lite/latest"
    else:
        model_uri = "gpt://"+str(yagpt_folder_id)+"/yandexgpt/latest"    
    llm = ChatYandexGPT(api_key=yagpt_api_key, model_uri=model_uri, temperature = yagpt_temperature, max_tokens = yagpt_max_tokens)
    # llm = YandexLLM(api_key = yagpt_api_key, folder_id = yagpt_folder_id, temperature = 0.6, max_tokens=8000, use_lite = False)

    # инициализация объекта класса YandexEmbeddings
    embeddings = YandexEmbeddings(folder_id=yagpt_folder_id, api_key=yagpt_api_key)
    # Поскольку эмбеддинги уже есть, то запускаем эту строчку
    vectorstore = OpenSearchVectorSearch (
        embedding_function=embeddings,
        index_name = mdb_os_index_name,
        opensearch_url=mdb_os_hosts,
        http_auth=("admin", mdb_os_pwd),
        use_ssl = True,
        verify_certs = False,
        engine = 'lucene',
        space_type="cosinesimil"
    )
    # space_type: "l2", "l1", "cosinesimil", "linf", "innerproduct"; default: "l2"

    # определение ретривера по векторной БД
    retriever = vectorstore.as_retriever(search_kwargs={"k": rag_k})

    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    question_answer_chain = create_stuff_documents_chain(llm, prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    ### Управление историей общения ###
    store = {}

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = StreamlitChatMessageHistory()
        return store[session_id]


    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )


    # Отображать текущие сообщения из StreamlitChatMessageHistory
    for msg in msgs.messages:
        st.chat_message(msg.type).write(msg.content)

    # Если пользователь вводит новое приглашение, сгенерировать и отобразить новый ответ
    if prompt := st.chat_input():
        st.chat_message("human").write(prompt)
        # Примечание: новые сообщения автоматически сохраняются в историю по длинной цепочке во время запуска
        config = {"configurable": {"session_id": "any"}}
        inputs = {"input": prompt}
        response = conversational_rag_chain.invoke(inputs,config)
        # response = chain_with_history.invoke({"question": prompt}, config)
        st.chat_message("ai").write(response["answer"])

        # Добавляем озвучку к ответу
        file_name = "./Reply.mp3"
        params = {"text": response["answer"],"voice": "zabelin"}
        res_tts = requests.get(sk_api_ep, params=params)
        # The response is a stream of bytes, so you can write it to a file
        with open(file_name, "wb") as f:
            f.write(res_tts.content)  

        # Добавляем озвучку к вопросу
        file_name = "./Question.mp3"
        params = {"text": prompt,"voice": "zabelin"}
        res_tts = requests.get(sk_api_ep, params=params)
        # The response is a stream of bytes, so you can write it to a file
        with open(file_name, "wb") as f:
            f.write(res_tts.content)          

        # Создаем две колонки
        col1, col2 = st.columns(2)
        # В первой колонке выводим "Озвучить ответ"
        with col1:
            st.write("Озвучить ответ:")
            st.audio("./Reply.mp3", format="audio/mpeg", loop=False)
        # Во второй колонке выводим "Озвучить вопрос"
        with col2:
            st.write("Озвучить вопрос:")
            st.audio("./Question.mp3", format="audio/mpeg", loop=False)

        ## добавляем источники к ответу
        input_documents = response["context"]
        i = 0
        for doc in input_documents:
            source = doc.metadata['source']
            page_content = doc.page_content
            i = i + 1
            with st.expander(f"**Источник N{i}:** [{source}]"):
                st.write(page_content)    


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