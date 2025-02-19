import streamlit as st
from langchain_mistralai.chat_models import ChatMistralAI
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationChain
from app.rag_utils import create_rag_prompt
from streamlit_chat import message


def start_messaging(rag_top_k=5, max_memory_size=4096):
    '''Функция для функционирования чата'''
    llm = ChatMistralAI(
        model="mistral-small-latest",
        api_key='#############',
        streaming=True
    )

    conversation = ConversationChain(
        llm=llm,
        memory=ConversationSummaryMemory(llm=llm),
        verbose=True
    )

    if "memory_size" not in st.session_state:
        st.session_state.memory_size = 0
    if "context_documents" not in st.session_state:
        st.session_state.context_documents = []
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "user_input" not in st.session_state:
        st.session_state.user_input = ''

    st.title("💬 Умный советник курсов 🤖")

    if st.button("Очистить чат"):
        st.session_state.messages = []
        st.session_state.memory_size = 0
        st.session_state.context_documents = []
        st.session_state.user_input = ''
        st.rerun()

    for i, message_data in enumerate(st.session_state.messages):
        if message_data['role'] == 'user':
            message(
                message_data['text'],
                is_user=True,
                key=f"user_{i}",
                avatar_style="thumbs",
                seed=i,
            )
        else:
            message(
                message_data['text'],
                is_user=False,
                key=f"model_{i}",
                avatar_style="bottts",
                seed=i,
            )

    def send_message():
        '''Функция для отправки запроса в LLM'''
        user_input = st.session_state.user_input
        
        st.session_state.messages.append({"role": "user", "text": user_input})

        prompt = create_rag_prompt(user_input, top_k=rag_top_k)

        new_memory = conversation.memory.load_memory_variables({})['history']
        new_memory_size = len(new_memory)

        try:

            if st.session_state.memory_size + new_memory_size <= max_memory_size:
                st.session_state.context_documents.append(new_memory)
                st.session_state.memory_size += new_memory_size
                response = conversation.predict(input=prompt)
            else:
                st.warning("Достигнут предел памяти, дальнейшее накопление прекращено.")
                response = conversation.predict(input=user_input)

            st.session_state.messages.append({"role": "bot", "text": response})
            st.session_state.user_input = ''
        except:
            st.warning('Попробуйте ещё раз')

    st.text_input(
        "Введите сообщение",
        placeholder="Например, какие курсы подходят для аналитиков данных?",
        key="user_input",
        on_change=send_message
    )


if __name__ == "__main__":
    start_messaging()
