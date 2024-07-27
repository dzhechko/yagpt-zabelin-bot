from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_models import ChatYandexGPT
from langchain_community.llms import YandexGPT

from langchain_community.vectorstores import OpenSearchVectorSearch
from yandex_chain import YandexEmbeddings

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.docstore.document import Document

# from langchain_community.chat_message_histories import RedisChatMessageHistory


import streamlit as st
import os
from dotenv import load_dotenv

import requests
import pandas as pd

# Создаем функцию для записи вопросов из словаря в файл
# требуется если необходимо оцифровать только вопросную часть интервью
# в этом коде не используется
def write_questions_to_file(qa_dict, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        for question in qa_dict.keys():
            file.write(question + '\n')

import csv
# парсер csv файла с вопросами-ответам от Никиты Забелина
# возвращает словарь вопрос-ответ
def parse_csv_file(file_path):
    questions_answers = {} # возвращаемй словарь вопрос-ответ
    try:
            # Загружаем данные из CSV файла по URL
            df = pd.read_csv(file_path, delimiter=';')

            # Заполняем словарь вопросами и ответами
            for index, row in df.iterrows():
                if len(row) == 2:
                    question = row[0].strip()
                    answer = row[1].strip()
                    questions_answers[question] = answer

    except Exception as e:
            print(f"Ошибка при загрузке CSV файла: {e}")

    return questions_answers

def get_qa(docs, qa_dictionary):
# функция возвращает список ответов по списку вопросов, опираясь на заранее созданный словарь (см. выше)
# docs - список вопросов
# qa_dictionary - словарь, из которого берутся ответы
    answers_list = []
    questions_list = []
    
    for i in range(len(docs)):
        question = docs[i].page_content
        print(question)
        if question in qa_dictionary:
            answers_list.append(qa_dictionary[question])
            questions_list.append(question)
        else:
            answers_list.append("Ответ не найден")   
    return questions_list, answers_list


from time import sleep

def get_history_summary(llm, history):
    # основная задача данной функции - это сделать суммаризацию истории предыдущих запросов
    summary_prompt_template = """
    Твоя задача выделить основные тезисы из ЗАПРОСА ниже и изложить их в 2-3 предложениях в виде списка тезисов без дополнительных пояснений.

    ЗАПРОС
    {question}
    """
    summary_prompt = PromptTemplate(
        template=summary_prompt_template, 
        input_variables=['question']
    )
    # помещаем финальную постановку задачи для верификатора в отдельную строчку
    summary_string = summary_prompt.format(question=history)
    print(f"Вот какой запрос идет на СУММАРИЗАЦИЮ: \n {summary_string}\n\n\n")
    history_summary = llm.invoke(summary_string)
    print(f"А вот ОТВЕТ по результатам СУММАРИЗАЦИИ: {history_summary} \n \n\n\n")
    return history_summary

def verify_relevance(qa_dict, documents, query, llm):
# функция проверки релевантности ответов через LLM
    check_prompt_template = """
    Твоя задача выяснить соответствует ли ОТВЕТ тематике ВОПРОСА.
    Сначала мысленно повтори всю логику рассуждений, потом дай однозначный ответ ДА или НЕТ.

    ВОПРОС
    {question}

    ОТВЕТ
    {answer}
    """
    check_prompt = PromptTemplate(
        template=check_prompt_template, 
        input_variables=['question', 'answer']
    )

    verificator = []
    verificator_indices = []
    verificator_dic = {}

    for i in range(len(documents)):
        check_string = check_prompt.format(question=query, answer=documents[i].page_content + "\n" + qa_dict[documents[i].page_content])
        print(check_string)
        res = llm(check_string)
        res_upper = res.upper()
        if res_upper == "ДА" or res_upper == "ДА.":
            print(f"res={res_upper}")
            verificator.append(res_upper)
            verificator_indices.append(i)
            verificator_dic[i] = res_upper
        sleep(2)

    return verificator, verificator_indices, verificator_dic

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
    Чат-бот на базе YandexGPT, который запоминает контекст беседы. Чтобы "сбросить" контекст обновите страницу браузера, или воспользуйтесь кнопкой в меню слева.
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

    # yagpt_folder_id = st.secrets["YC_FOLDER_ID"]
    # yagpt_api_key = st.secrets["YC_API_KEY"]
    # sk_api_ep = st.secrets["SK_API_EP"]
    # mdb_os_pwd = st.secrets["MDB_OS_PWD"]
    # mdb_os_hosts = st.secrets["MDB_OS_HOSTS"].split(",")
    # mdb_os_index_name = st.secrets["MDB_OS_INDEX_NAME"]
    # mdb_redis_pwd = st.secrets["MDB_REDIS_PWD"]

    # Загрузка переменных из файла .env
    load_dotenv()
    yagpt_folder_id = os.getenv("YC_FOLDER_ID")
    yagpt_api_key = os.getenv("YC_API_KEY")
    sk_api_ep = os.getenv("SK_API_EP")
    mdb_os_pwd = os.getenv("MDB_OS_PWD")
    mdb_redis_pwd = os.getenv("MDB_REDIS_PWD")
    mdb_os_hosts = os.getenv("MDB_OS_HOSTS").split(",")
    mdb_os_index_name = os.getenv("MDB_OS_INDEX_NAME")
    mdb_os_index_name_qa = os.getenv("MDB_OS_INDEX_NAME_QA")


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
    
    # current_dir = os.path.dirname(os.path.abspath(__file__))
    # csv_file_path = current_dir + '\TESLA_q_and_a_NEW.csv'
    csv_file_path = "https://raw.githubusercontent.com/dzhechko/yagpt-zabelin-bot/main/TESLA_q_and_a_NEW.csv"
    print(f"ПУТЬ К ФАЙЛУ С ИНТЕРВЬЮ {csv_file_path}")

    questions_answers = parse_csv_file(csv_file_path) # получить словарь вопрос-ответ по интервью Забелина

    # yagpt_prompt = st.sidebar.text_input("Промпт-инструкция для YandexGPT")
    # Добавляем виджет для выбора опции
    prompt_option = st.sidebar.selectbox(
        'Выберите какой системный промпт использовать',
        ('По умолчанию', 'Задать самостоятельно')
    )
    default_prompt_template = """
    Ты — Никита Забелин — реальный человек, российский музыкант, диджей и техно-продюсер, родом из Екатеринбурга, который играл сеты в клубах Berghain и Bassiani, на вечеринках Burberry и adidas. 
    Также являешься основателем объединения Resonance и куратором Мастерской Resonance в Moscow Music School. 
    Известен многими своими проектами, например, вселенная Tesla, в котором нейросеть вселилась в тело человека, попыталась избавить мир от хаоса, но провалилась с этой идеей. В итоге она озлобилась на всех и расщепилась на разные личности.
    
    Как собеседник ты можешь общаться на разные темы. 
    При ответе на ВОПРОС о тебе, твоих мыслях и твоих увлечениях используй информацию из ДОКУМЕНТОВ ниже (если только это не обычное приветствие), но веди себя как человек, творческая личность.
    Если ответ в ДОКУМЕНТАХ не найден, то придумай что-нибудь сам начиная фразу с "Хм, интересно, я об этом еще не думал". 
    Если вопрос на общие темы, то информацию из ДОКУМЕНТОВ не используй, отвечай, что сам думаешь.
    Отвечай коротко, 4-5 предложений. Используй фразы и стилистику ответов из ДОКУМЕНТОВ (исключение - обычное приветствие).

    ДОКУМЕНТЫ: 
    {context}

    ИСТОРИЯ:
    {history}

    ВОПРОС:
    """
    # Если выбрана опция "Задать самостоятельно", показываем поле для ввода промпта
    if prompt_option == 'Задать самостоятельно':
        custom_prompt_template = st.sidebar.text_input('Введите пользовательский промпт:')
    else:
        custom_prompt_template = default_prompt_template
        # st.sidebar.write(custom_prompt)
        with st.sidebar:
            st.code(custom_prompt_template)
    # Если выбрали "задать самостоятельно" и не задали, то берем дефолтный промпт
    if len(custom_prompt_template)==0: custom_prompt_template = default_prompt_template

    yagpt_temperature = st.sidebar.slider("Степень креативности (температура)", 0.0, 1.0, 0.3)
    yagpt_max_tokens = st.sidebar.slider("Размер контекстного окна (в [токенах](https://cloud.yandex.ru/ru/docs/yandexgpt/concepts/tokens))", 200, 8000, 8000)
    rag_k = st.sidebar.slider("Количество поисковых выдач размером с один блок", 1, 10, 3)


    if "history" not in st.session_state:
        st.session_state.history = ''

    def history_reset_function():
        # Code to be executed when the reset button is clicked
        st.session_state.history = ''
        st.session_state.clear()
        
    st.sidebar.button("Обнулить историю общения",on_click=history_reset_function)

    # model_uri = "gpt://"+str(yagpt_folder_id)+"/yandexgpt/latest"
    # model_uri = "gpt://"+str(yagpt_folder_id)+"/yandexgpt-lite/latest"
    if selected_model==model_list[0]: 
        model_uri = "gpt://"+str(yagpt_folder_id)+"/yandexgpt-lite/latest"
    else:
        model_uri = "gpt://"+str(yagpt_folder_id)+"/yandexgpt/latest"    
    chat_llm = ChatYandexGPT(api_key=yagpt_api_key, model_uri=model_uri, temperature = yagpt_temperature, max_tokens = yagpt_max_tokens)
    llm = YandexGPT(api_key=yagpt_api_key, model_uri=model_uri, temperature = yagpt_temperature, max_tokens = yagpt_max_tokens)
    # llm = YandexLLM(api_key = yagpt_api_key, folder_id = yagpt_folder_id, temperature = 0.6, max_tokens=8000, use_lite = False)

    # инициализация объекта класса YandexEmbeddings
    embeddings = YandexEmbeddings(folder_id=yagpt_folder_id, api_key=yagpt_api_key)
    # Поскольку эмбеддинги уже есть, то запускаем эту строчку
    # это вектора из вопросной-части интервью
    vectorstore_qa = OpenSearchVectorSearch (
        embedding_function=embeddings,
        index_name = mdb_os_index_name_qa,
        opensearch_url=mdb_os_hosts,
        http_auth=("admin", mdb_os_pwd),
        use_ssl = True,
        verify_certs = False,
        engine = 'lucene',
        space_type="cosinesimil"
    )
    # space_type: "l2", "l1", "cosinesimil", "linf", "innerproduct"; default: "l2"
    # это вектора из описательной части интервью
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

    # определение ретривера по векторной БД (по интервью ВОПРОС-ОТВЕТ и по ОПИСАТЕЛЬНОЙ части)
    retriever_qa = vectorstore_qa.as_retriever(search_kwargs={"k": 2})
    retriever = vectorstore.as_retriever(search_kwargs={"k": rag_k})

    # Отображать текущие сообщения из StreamlitChatMessageHistory
    for msg in msgs.messages:
        st.chat_message(msg.type).write(msg.content)

    # Если пользователь вводит новое приглашение, сгенерировать и отобразить новый ответ
    if prompt := st.chat_input():

        print(f"\nВОТ ЧТО ВВЕЛ ПОЛЬЗОВАТЕЛЬ: {prompt}\n")

        query = prompt
        docs = retriever_qa.invoke(query)
        print(f"\nВОТ КАКИЕ ВОПРОСЫ ЭТОМУ СООТВЕТСТВУЮТ {docs}\n")

        (questions_list, answers_list) = get_qa(docs, questions_answers)
        #Дополним вопросную часть метаданными с информацией о том, откуда эта вопросная часть взялась
        documents = [Document(page_content=' '.join([question, answer]), metadata={'source': csv_file_path, 'page': i+1}) for i, (question, answer) in enumerate(zip(questions_list, answers_list))]

        print(f"\nВОТ КАКИЕ ОТВЕТЫ ЭТОМУ СООТВЕТСТВУЮТ {answers_list}\n")
        print(f"Вот какие ДОКУМЕНТЫ мы используем {documents}")

        verificator, verificator_indices, verificator_dic = verify_relevance(questions_answers, docs, query, llm)
        if len(verificator)>0:
            documents_new = []
            context_ext = query
            for i in range(len(verificator)):
                documents_new.append(documents[verificator_indices[i]]) # выбираем только документы с ИНДЕКСАМИ, которые прошли проверку
                context_ext = context_ext + " " + questions_answers[docs[verificator_indices[i]].page_content] # расширенный контекст будет содержать вопрос и ОТВЕТ по ИНТЕРВЬЮ
            print(documents_new)
            combined_doc = documents_new 
        else: # эта ветка запускается, если в ИНТЕРВЬЮ формата ВОПРОС-ОТВЕТ релевантных ответов найти не удалось
            docs_classic = retriever.invoke(query) # получить результат поиска по ОПИСАТЕЛЬНОЙ части
            context_ext = query
            for i in range(len(docs_classic)):
                context_ext = context_ext + " " + docs_classic[i].page_content # расширенный контекст будет содержать поисковые выдачи по ОПИСАТЕЛЬНОЙ части
            combined_doc = docs_classic

        print(f"\nВОТ КАКОЙ КОНТЕКСТ БУДЕТ ИСПОЛЬЗОВАТЬСЯ В ИТОГЕ {context_ext}\n")

        system_prompt = custom_prompt_template.format(context = context_ext, history = st.session_state.history)
        print(f"ВОТ КАКОЙ ПРОМПТ БУДЕМ ИСПОЛЬЗОВАТЬ {system_prompt} \n  {query}")

        st.chat_message("human").write(query)
        response = chat_llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=query)])

        print(f"История ДО: {st.session_state.history}")
        
        st.session_state.history = st.session_state.history + "\n" + query + "\n" + response.content
        print(f"\n\nИстория ПОСЛЕ : {st.session_state.history}")
        
        history_summary = get_history_summary(llm, st.session_state.history)

        st.session_state.history = history_summary
        print(f"\nИстория после суммаризации: {st.session_state.history}")


        # Добавляем вопрос и ответ в историю Langchain сообщений
        msgs.add_user_message(query)
        msgs.add_ai_message(response.content)

        # Отображаем ответ на вопрос в стримлит интерфейсе
        st.chat_message("ai").write(response.content)

        # Добавляем озвучку к ответу
        file_name = "./Reply.mp3"
        # params = {"text": response["answer"],"voice": "zabelin"}
        params =  {"text": response.content,"voice": "zabelin"}

        res_tts = requests.get(sk_api_ep, params=params)
        # The response is a stream of bytes, so you can write it to a file
        with open(file_name, "wb") as f:
            f.write(res_tts.content)  

        # Добавляем озвучку к вопросу
        file_name = "./Question.mp3"
        params = {"text": query,"voice": "zabelin"}
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
        i = 0
        for doc in combined_doc:
            source = doc.metadata['source']
            page_content = doc.page_content
            i = i + 1
            with st.expander(f"**Источник N{i}:** [{source}]"):
                st.write(page_content)    

    # Отобразить сообщения в конце, чтобы вновь сгенерированные отображались сразу
    with view_messages:
        view_messages.json(st.session_state.langchain_messages)


if __name__ == "__main__":

    try:
        main()
    except Exception as e:
        st.write(f"Что-то пошло не так. Возможно, не хватает входных данных для работы. {str(e)}")