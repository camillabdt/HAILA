import time
import json
import pandas as pd
from groq import Groq
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import os
import re

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

print("🔧 Script iniciado")
print(f"🔐 GROQ_API_KEY carregada: {'sim' if GROQ_API_KEY else 'não'}")

# Modelos que você está usando
LLMS = {
    "gpt-oss-120b": "openai/gpt-oss-120b",
    "qwen3-32b": "qwen/qwen3-32b",
    "compound": "groq/compound",
    "llama-3-70b": "llama-3.1-8b-instant",
}

# Carrega as questões geradas
with open("questoes.json", "r", encoding="utf-8") as f:
    questoes = json.load(f)

print(f"✅ {len(questoes)} questões carregadas")

def gerar_prompt_avaliacao(questao):
    """
    Gera um prompt para avaliar a questão usando as MESMAS 5 métricas
    que o professor vai usar na interface
    """
    resposta = questao.get("response", "")
    
    return f"""
Você é um avaliador especializado em questões geradas por modelos de linguagem.

A seguir está uma resposta gerada por um modelo. Avalie-a atribuindo uma nota de **1 a 5** para cada um dos critérios descritos abaixo, **sem justificar**. Utilize apenas o formato solicitado.

---

Resposta: {resposta}

---

Avalie conforme os seguintes critérios:

1. Fluidez Gramatical — Clareza, correção linguística e coesão do texto.
2. Capacidade de Resposta — Se a resposta realmente responde bem à pergunta/tarefa proposta.
3. Diversidade — Variedade de perspectivas e abordagens apresentadas na resposta.
4. Complexidade — Nível de raciocínio e profundidade exigido para gerar essa resposta.
5. Relevância — Alinhamento da resposta com o tema e contexto proposto.

Retorne sua resposta **exclusivamente no seguinte formato** (minúsculas, sem comentários):

fluidez: X
capacidade: X
diversidade: X
complexidade: X
relevancia: X

Onde X é um número de 1 a 5.
""".strip()

def extrair_notas(resposta_texto):
    """Extrai as 5 notas do formato esperado"""
    notas = {}
    
    # Padrões para cada métrica
    padroes = {
        "fluidez": r"(?i)fluidez\s*:\s*([1-5])",
        "capacidade": r"(?i)capacidade\s*:\s*([1-5])",
        "diversidade": r"(?i)diversidade\s*:\s*([1-5])",
        "complexidade": r"(?i)complexidade\s*:\s*([1-5])",
        "relevancia": r"(?i)relev[aâ]ncia\s*:\s*([1-5])",
    }
    
    for metrica, padrao in padroes.items():
        match = re.search(padrao, resposta_texto)
        if match:
            notas[metrica] = int(match.group(1))
    
    return notas

def avaliar_questao(avaliador_nome, modelo_id, questao, tentativas=3):
    """
    Avalia uma questão usando um modelo específico
    Evita auto-avaliação (um modelo não avalia suas próprias questões)
    """
    modelo_gerador = questao.get("llm", "").lower()
    
    # Evita auto-avaliação
    if avaliador_nome.lower() in modelo_gerador.lower():
        return None
    
    prompt = gerar_prompt_avaliacao(questao)
    
    for tentativa in range(tentativas):
        try:
            resposta = client.chat.completions.create(
                model=modelo_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=1000
            )
            content = resposta.choices[0].message.content.strip()
            notas = extrair_notas(content)
            
            # Verifica se todas as 5 métricas foram extraídas
            if len(notas) == 5:
                return {
                    "questao_numero": questao.get("question_number", ""),
                    "avaliador": avaliador_nome,
                    "modelo_avaliado": questao.get("llm", ""),
                    "tecnica_prompt": questao.get("prompt_technique", ""),
                    "fluidez": notas.get("fluidez", 0),
                    "capacidade": notas.get("capacidade", 0),
                    "diversidade": notas.get("diversidade", 0),
                    "complexidade": notas.get("complexidade", 0),
                    "relevancia": notas.get("relevancia", 0),
                    "media": sum(notas.values()) / 5
                }
            else:
                print(f"⚠️ Notas incompletas (Questão {questao.get('question_number', '')}): {len(notas)}/5")
                print(f"   Resposta: {content[:100]}...\n")
                return None
                
        except Exception as e:
            if "rate_limit" in str(e).lower():
                print(f"⏳ Rate limit com {avaliador_nome}. Esperando 10s... ({tentativa + 1}/{tentativas})")
                time.sleep(10)
            else:
                print(f"❌ Erro com {avaliador_nome} ao avaliar Questão {questao.get('question_number', '')}: {e}")
                return None
    
    return None

# Executa as avaliações em paralelo
print("\n🔍 Iniciando avaliações cruzadas...")
print(f"📊 Total de avaliações: {len(LLMS)} avaliadores × {len(questoes)} questões")

avaliacoes = []
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = []
    
    for avaliador_nome, modelo_id in LLMS.items():
        for questao in questoes:
            futures.append(
                executor.submit(avaliar_questao, avaliador_nome, modelo_id, questao)
            )
    
    for future in tqdm(as_completed(futures), total=len(futures), desc="🔍 Avaliando"):
        resultado = future.result()
        if resultado:
            avaliacoes.append(resultado)

# Salva os resultados em CSV
print(f"\n✅ {len(avaliacoes)} avaliações completadas")

if avaliacoes:
    df_avaliacoes = pd.DataFrame(avaliacoes)
    
    # Salva em CSV
    df_avaliacoes.to_csv("avaliacoes_llms_cruzadas.csv", index=False, encoding="utf-8")
    print("📁 Arquivo salvo: avaliacoes_llms_cruzadas.csv")
    
    # Exibe resumo
    print("\n📊 Resumo das Avaliações:")
    print(f"Total de avaliações: {len(df_avaliacoes)}")
    print(f"\nMédia por métrica:")
    print(f"  Fluidez: {df_avaliacoes['fluidez'].mean():.2f}")
    print(f"  Capacidade: {df_avaliacoes['capacidade'].mean():.2f}")
    print(f"  Diversidade: {df_avaliacoes['diversidade'].mean():.2f}")
    print(f"  Complexidade: {df_avaliacoes['complexidade'].mean():.2f}")
    print(f"  Relevância: {df_avaliacoes['relevancia'].mean():.2f}")
    
    print(f"\nAvaliações por avaliador:")
    print(df_avaliacoes['avaliador'].value_counts())
    
    print(f"\nAvaliações por modelo avaliado:")
    print(df_avaliacoes['modelo_avaliado'].value_counts())
else:
    print("❌ Nenhuma avaliação foi completada!")
