import streamlit as st
import torch
from langchain.embeddings import HuggingFaceInstructEmbeddings
from transformers import AutoTokenizer, TextStreamer, pipeline
from auto_gptq import AutoGPTQForCausalLM
from langchain import HuggingFacePipeline, PromptTemplate
from streamlit_chat import message


st.title("chatbot")
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
embeddings = HuggingFaceInstructEmbeddings(
    model_name="hkunlp/instructor-large", model_kwargs={"device": DEVICE}
)

### PDF LOADER AND VECTOR STORE
from langchain_community.document_loaders import PyPDFLoader
pdf_name = './samsung.pdf'
loader = PyPDFLoader(pdf_name)
docs = loader.load_and_split()
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=64)
texts = text_splitter.split_documents(docs)
from langchain.vectorstores import Chroma
db = Chroma.from_documents(texts, embeddings, persist_directory="db")


### MODEL
model_name_or_path = "TheBloke/Llama-2-7B-chat-GPTQ"
model_basename = "model"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

model = AutoGPTQForCausalLM.from_quantized(
    model_name_or_path,
    revision="gptq-4bit-128g-actorder_True",
    model_basename=model_basename,
    use_safetensors=True,
    trust_remote_code=True,
    inject_fused_attention=False,
    device=DEVICE,
    quantize_config=None,
    disable_exllama=True,
    auto_devices=True
)

streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

text_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=1024,
    temperature=0,
    top_p=0.95,
    repetition_penalty=1.15,
    streamer=streamer
    )

llm_model = HuggingFacePipeline(pipeline=text_pipeline, model_kwargs={"temperature": 0})



def generate_prompt(prompt: str, system_prompt: str) -> str:
    return f"""
        [INST] <<SYS>>
        {system_prompt}
        <</SYS>>

        {prompt} [/INST]
        """.strip()

SYSTEM_PROMPT = "Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer."

template = generate_prompt(
    """
{context}

Question: {question}
""",
    system_prompt=SYSTEM_PROMPT,
)
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
memory = ConversationBufferMemory(
    memory_key="chat_history", output_key='answer', return_messages=False)

prompt = PromptTemplate(template=template, input_variables=["context", "question"])
chain = ConversationalRetrievalChain.from_llm(
    llm=llm_model,
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 1}),
    get_chat_history=lambda o:o,
    memory=memory,
    return_generated_question=True,
    verbose=False,
    combine_docs_chain_kwargs={"prompt": prompt}
)


def conversational_chat(query):
    result = chain({"question": query, "chat_history": st.session_state['history']})
    st.session_state['history'].append((query, result["answer"]))
    return result["answer"]

# Initialize chat history
if 'history' not in st.session_state:
    st.session_state['history'] = []

# Initialize messages
if 'generated' not in st.session_state:
    st.session_state['generated'] = [f"Hello ! Ask me(LLAMA2) about this {pdf_name} PDF 🤗"]

if 'past' not in st.session_state:
    st.session_state['past'] = ["Hey ! 👋"]

response_container = st.container()
container = st.container()

# User input form
with container:
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_input("Query:", placeholder="Talk to csv data 👉 (:", key='input')
        submit_button = st.form_submit_button(label='Send')

    if submit_button and user_input:
        output = conversational_chat(user_input)
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(output)

# Display chat history
if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
            message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")