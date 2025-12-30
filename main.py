import sys
import os

# Streamlit Cloud(Linux)에서만 pysqlite3 사용
if os.environ.get("STREAMLIT_CLOUD") == "true":
    try:
        __import__("pysqlite3")
        sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
    except ModuleNotFoundError:
        pass

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_classic.chains import RetrievalQA

import streamlit as st
import tempfile
import os
import hashlib

# =================================================
# 기본 설정
# =================================================
load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PERSIST_DIR = os.path.join(BASE_DIR, "chroma_langchain_db")

# =================================================
# Streamlit UI Title
# =================================================
st.title("📄 ChatPDF")
st.write("---")

# =================================================
# OPENAI_API_KEY AI 키 입력 받고,
# 환경 변수 등록, 하위 OpenAI 관련 API는 냅둬도됨. 
# =================================================
openai_key= st.text_input("OPENAI_API_KEY", type="password")

if openai_key:
    os.environ["OPENAI_API_KEY"] = openai_key

# =================================================
# Streamlit UI File Upload
# =================================================

uploaded_file = st.file_uploader("PDF 파일을 업로드하세요", type=["pdf"])
st.write("---")

# =================================================
# Utils
# =================================================
def file_hash(uploaded_file) -> str:
    return hashlib.sha256(uploaded_file.getvalue()).hexdigest()

@st.cache_resource
def load_vectorstore(collection_name: str):
    embeddings = OpenAIEmbeddings()
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )

# =================================================
# PDF 처리
# =================================================
if uploaded_file is not None:
    file_id = file_hash(uploaded_file)
    collection_name = f"pdf_{file_id[:12]}"

    vector_store = load_vectorstore(collection_name)

    # ✅ 임베딩 스피너 (최초 1회만)
    if vector_store._collection.count() == 0:
        with st.spinner("📌 처음 업로드된 PDF입니다. 임베딩 중입니다..."):
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = os.path.join(temp_dir, uploaded_file.name)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getvalue())

                loader = PyPDFLoader(temp_path)
                documents = loader.load()

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=300,
                    chunk_overlap=20
                )
                split_docs = splitter.split_documents(documents)

                vector_store.add_documents(split_docs)
                try:
                    vector_store.persist()
                except Exception:
                    pass

        st.success("✅ 임베딩 완료!")
    else:
        st.success("✅ 기존 벡터 DB 재사용")

    st.write("---")

    # Retriever / QA
    llm = ChatOpenAI(temperature=0, max_completion_tokens=2048)
    base_retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    mqr = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm,
        include_original=True
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=mqr,
        return_source_documents=True
    )

    # 문서 요약
    if st.button("📌 문서 요약"):
        with st.spinner("🧠 문서 요약 생성 중..."):
            result = qa.invoke({"query": "이 문서 핵심 요약해줘"})
        st.subheader("📌 요약")
        st.write(result["result"])

    st.write("---")

    # PDF에게 질문해보세요
    st.subheader("🤖 PDF에게 질문해보세요")
    user_question = st.text_input(
        label="",
        placeholder="PDF 내용에 대해 궁금한 점을 입력하세요",
    )

    if user_question and st.button("질문하기"):
        with st.spinner("🔎 문서를 찾고 답변을 생성하는 중..."):
            result = qa.invoke({"query": user_question})

        st.subheader("💬 답변")
        st.write(result["result"])

        st.subheader("📎 근거 문서 (페이지)")
        for i, doc in enumerate(result["source_documents"], 1):
            st.markdown(f"**[{i}] page {doc.metadata.get('page')}**")
            st.text(doc.page_content[:300])
else:
    st.info("PDF를 업로드하면 질문 입력창이 나타납니다.")
