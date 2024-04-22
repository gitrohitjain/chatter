# import os
import torch
# import json
# import re
from auto_gptq import AutoGPTQForCausalLM
# from langchain.schema.document import Document
from langchain import HuggingFacePipeline, PromptTemplate
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, TextStreamer, pipeline
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
# from google.cloud import aiplatform
# from langchain.chat_models import ChatVertexAI



DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
embeddings = HuggingFaceInstructEmbeddings(
    model_name="hkunlp/instructor-large", model_kwargs={"device": DEVICE}
)
print('DEVICE:', DEVICE)

from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader("./samsung.pdf")
docs = loader.load_and_split()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=64)
texts = text_splitter.split_documents(docs)

db = Chroma.from_documents(texts, embeddings, persist_directory="db")

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

memory = ConversationBufferMemory(
    memory_key="chat_history", output_key='answer', return_messages=False)

prompt = PromptTemplate(template=template, input_variables=["context", "question"])
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm_model,
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 1}),
    get_chat_history=lambda o:o,
    memory=memory,
    return_generated_question=True,
    verbose=False,
    combine_docs_chain_kwargs={"prompt": prompt}
)

result = qa_chain("How to power off the phone?")
print(result['answer'])