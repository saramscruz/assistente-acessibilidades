"""
Versão LEVE, para publicar online. Usa um modelo de embeddings pequeno
(~470 MB) para caber no limite de memória gratuito do Streamlit Cloud.

Correr no SEU computador, uma vez (ou sempre que o JSON mudar):
    pip install chromadb sentence-transformers --break-system-packages
    python indexar_chroma_leve.py
"""
import json
import shutil
import os
import chromadb
from sentence_transformers import SentenceTransformer

MODELO = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
modelo = SentenceTransformer(MODELO)

with open("dl_163_2006_estruturado.json", "r", encoding="utf-8") as f:
    dados = json.load(f)

colecao_dados = []
for a in dados["artigos"]:
    colecao_dados.append({
        "id": f"art_{a['numero']}",
        "texto": a["texto"],
        "citacao": f"{dados['diploma']}, Artigo {a['numero']} ({a['epigrafe']})",
    })
for it in dados["anexo_itens"]:
    colecao_dados.append({
        "id": f"anexo_{it['ref']}",
        "texto": it["texto"],
        "citacao": f"{dados['diploma']}, Anexo, ponto {it['ref']} ({it['capitulo_titulo']} > {it['seccao_titulo']})",
    })

# apagar a base antiga para garantir que fica reconstruída com o espaço de
# distância certo (mudar de métrica numa base já existente não é seguro)
if os.path.exists("./base_modulos_leve"):
    shutil.rmtree("./base_modulos_leve")

cliente = chromadb.PersistentClient(path="./base_modulos_leve")
modulo_acessibilidades = cliente.get_or_create_collection(
    "acessibilidades",
    metadata={"modelo": MODELO, "hnsw:space": "cosine"},  # distância explícita
)

embeddings = modelo.encode([c["texto"] for c in colecao_dados], normalize_embeddings=True).tolist()
modulo_acessibilidades.add(
    ids=[c["id"] for c in colecao_dados],
    embeddings=embeddings,
    documents=[c["texto"] for c in colecao_dados],
    metadatas=[{"citacao": c["citacao"]} for c in colecao_dados],
)

print(f"Módulo 'acessibilidades' indexado com {len(colecao_dados)} entradas (distância: cosine).")

# --- teste rápido com as perguntas reais que falharam, para ver as distâncias ---
testes = [
    "largura mínima de uma rampa de acesso",
    "dimensões mínimas de uma casa de banho acessível",
    "largura mínima de uma escada num prédio de apartamentos",
]
for p in testes:
    vec = modelo.encode([p], normalize_embeddings=True).tolist()
    r = modulo_acessibilidades.query(query_embeddings=vec, n_results=2)
    print(f"\nPERGUNTA: {p}")
    for doc, meta, dist in zip(r["documents"][0], r["metadatas"][0], r["distances"][0]):
        print(f"  distância={dist:.3f} | {meta['citacao']}")
