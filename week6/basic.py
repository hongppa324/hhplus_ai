import os
import base64
import streamlit as st

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="ImaGePT", layout="wide")
st.title("🖼️ ImaGePT")

# 모델 초기화
model = ChatOpenAI(model="gpt-4o-mini")

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []
if "encoded_image" not in st.session_state:
    st.session_state.encoded_image = None
if "user_question" not in st.session_state:
    st.session_state.user_question = ""

# 폼 제출 처리 함수
def handle_submit():
    if st.session_state.user_input.strip():
        st.session_state.user_question = st.session_state.user_input
        st.session_state.submit_clicked = True
    
# 입력창 초기화 함수
def clear_text():
    st.session_state.user_input = ""

# CSS 스타일 추가
st.markdown(
    """
    <style>
    .chat-container {
        height: 60vh;
        overflow-y: auto;
        display: flex;
        flex-direction: column-reverse;
        padding-bottom: 80px;
    }
    .input-container {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background-color: white;
        padding: 10px 20px;
        box-shadow: 0 -2px 5px rgba(0,0,0,0.1);
        z-index: 9999;
    }
    .stButton button {
        background-color: #000;
        color: white;
        border: none;
        border-radius: 20px;
        width: 32px;
        height: 20px;
    }
    .image-preview {
        margin-bottom: 20px;
        max-height: 300px;
        text-align: center;
    }
    .main-content {
        margin-bottom: 100px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# 화면을 두 컬럼으로 분할
col1, col2 = st.columns([1, 2])

with col1:
    # 이미지 업로드 섹션
    st.subheader("이미지 업로드")
    image_list = st.file_uploader("어떤 이미지든 올려주세요!", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)
    
    if image_list:
        st.session_state.encoded_image = []  # 이미지 리스트 초기화

        # 2개씩 반복
        for i in range(0, len(image_list), 2):
            cols = st.columns(2)  # 두 개의 column 생성

            for j in range(2):
                if i + j < len(image_list):
                    with cols[j]:
                        image = image_list[i + j]
                        st.image(image, caption=f"업로드한 이미지 {i + j + 1}", use_container_width=True)
                        encoded = base64.b64encode(image.read()).decode("utf-8")
                        st.session_state.encoded_image.append(encoded)

with col2:
    st.subheader("이미지에 대해 질문하기")

    # 채팅 내역 표시 (과거 → 최신 순)
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)

    if st.session_state.messages:
        for q, a in st.session_state.messages:
            with st.chat_message("user"):
                st.markdown(q)
            with st.chat_message("assistant"):
                st.markdown(a)

    st.markdown('</div>', unsafe_allow_html=True)

    # 입력창과 입력 버튼
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    input_col1, input_col2 = st.columns([9, 1])

    with input_col1:
        user_question = st.text_input(
            "", placeholder="질문을 입력해주세요", label_visibility="collapsed", key="user_input", on_change=handle_submit
        )

    with input_col2:
        submit_button = st.button("⬆", use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # 입력 버튼 클릭 시
if submit_button:
    encoded_images = st.session_state.encoded_image

    if not encoded_images:
        st.warning("이미지를 먼저 업로드해주세요.")
    elif not user_question.strip():
        st.warning("질문을 입력해주세요.")
    else:
        with st.spinner("답변 생성 중..."):
            content = [{"type": "text", "text": user_question}]
            for encoded in encoded_images:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}
                })

            message = HumanMessage(content=content)
            result = model.invoke([message])
            response = result.content

        st.session_state.messages.append((user_question, response))
        st.rerun()

