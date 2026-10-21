import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="Assistente de Acessibilidades", page_icon="🏛️")

MODELO = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

@st.cache_resource
def carregar():
    modelo = SentenceTransformer(MODELO)
    cliente = chromadb.PersistentClient(path="./base_modulos_leve")
    colecao = cliente.get_collection("acessibilidades")
    return modelo, colecao

modelo, colecao = carregar()

st.title("🏛️ Assistente de Acessibilidades")
st.caption("Módulo: DL 163/2006 — respostas sempre com citação do artigo ou ponto do anexo")

pergunta = st.text_input("Faça uma pergunta sobre acessibilidades:",
                          placeholder="ex.: largura mínima de uma rampa")

if pergunta:
    vec = modelo.encode([pergunta]).tolist()
    resultados = colecao.query(query_embeddings=vec, n_results=4)
    docs = resultados["documents"][0]
    metas = resultados["metadatas"][0]
    distancias = resultados["distances"][0]

    if distancias[0] > 0.75:
        st.warning(
            "Não encontrei, com confiança, uma disposição que cubra isto no "
            "módulo de acessibilidades. Consulte directamente o diploma ou "
            "experimente reformular a pergunta."
        )
    else:
        st.subheader("Encontrado:")
        for doc, meta in zip(docs, metas):
            with st.container(border=True):
                st.markdown(f"**{meta['citacao']}**")
                st.write(doc)

st.divider()
st.caption(
    "⚠️ Ferramenta apenas informativa. Não substitui a verificação directa "
    "do diploma nem o aconselhamento profissional."
)
