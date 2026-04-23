import json
import time
from groq import Groq
from dotenv import load_dotenv
import os

# Carrega variáveis de ambiente
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

print("🔧 Script iniciado")
print(f"🔐 GROQ_API_KEY carregada: {'sim' if GROQ_API_KEY else 'não'}")

LLMS = {
"openai/gpt-oss-120b": "openai/gpt-oss-120b",
"qwen/qwen3-32b": "qwen/qwen3-32b",
"groq/compound": "groq/compound",
"llama-3-70b": "llama-3.1-8b-instant",}

# Instrução clara no final dos prompts
INSTRUCAO_FINAL = "\n\nInclua no final: a pergunta, quatro alternativas (A, B, C, D) e a resposta correta. Use o formato:\nPergunta: ...\nA) ...\nB) ...\nC) ...\nD) ...\nResposta correta: ..."

PROMPT_TEMPLATES = {
    "zero-shot": (
        "Gere uma questão de múltipla escolha (com 4 alternativas e apenas uma correta) "
        "sobre o tema 'Engenharia de Requisitos'." + INSTRUCAO_FINAL
    ),
    "few-shot": (
        "Exemplo 1:\n"
        "Pergunta: Qual é a principal finalidade da Engenharia de Requisitos em um projeto de software?\n"
        "A) Desenvolver o código-fonte\nB) Reduzir custos de hardware\nC) Definir e documentar as necessidades do cliente\nD) Realizar testes automatizados\n"
        "Resposta correta: C\n\n"
        "Exemplo 2:\n"
        "Pergunta: Qual dos itens a seguir é um requisito funcional?\n"
        "A) O sistema deve estar disponível 99,9% do tempo\nB) O sistema deve permitir login de usuários cadastrados\nC) A interface deve ser intuitiva\nD) O sistema deve responder em menos de 2 segundos\n"
        "Resposta correta: B\n\n"
        "Agora, gere uma nova questão sobre 'Engenharia de Requisitos'." + INSTRUCAO_FINAL
    ),
    "chain-of-thought": (
        "Pense passo a passo antes de responder.\n"
        "1. Considere os objetivos da Engenharia de Requisitos.\n"
        "2. Reflita sobre os tipos de requisitos e suas classificações.\n"
        "3. Baseando-se nesse raciocínio, gere uma questão de múltipla escolha (4 alternativas, uma correta) sobre Engenharia de Requisitos." + INSTRUCAO_FINAL
    ),
    "exemplar-guided": (
        "Exemplo (baseado em práticas da Engenharia de Requisitos):\n"
        "Pergunta: Durante a elicitação de requisitos, qual técnica é mais indicada para compreender o fluxo de trabalho atual do cliente?\n"
        "A) Entrevistas com usuários\nB) Análise de código\nC) Testes de unidade\nD) Refatoração de código\n"
        "Resposta correta: A\n\n"
        "Agora, baseado nesse estilo de questão, gere uma nova pergunta sobre Engenharia de Requisitos." + INSTRUCAO_FINAL
    ),
    "template-based": (
        "Complete o seguinte template de questão sobre Engenharia de Requisitos:\n"
        "Pergunta: [Enunciado sobre um conceito, técnica ou classificação da Engenharia de Requisitos]\n"
        "A) [Alternativa errada]\nB) [Alternativa errada]\nC) [Alternativa correta]\nD) [Alternativa errada]\n"
        "Resposta correta: [Letra da correta]\n\n"
        "Preencha com conteúdo coerente sobre o tema." + INSTRUCAO_FINAL
    )
}

TECHNIQUES = list(PROMPT_TEMPLATES.keys())
N_QUESTIONS = 10

def query_groq(model: str, prompt: str) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,  # mais determinístico
        max_tokens=1024,
        top_p=1
    )
    return response.choices[0].message.content.strip()

results = []

for llm_name, model_id in LLMS.items():
    for technique in TECHNIQUES:
        for i in range(1, N_QUESTIONS + 1):
            prompt = PROMPT_TEMPLATES[technique]
            try:
                response = query_groq(model_id, prompt)
            except Exception as e:
                response = f"[ERRO] {str(e)}"
            result = {
                "id": f"{llm_name.lower()}_{technique}_{i}",
                "llm": llm_name.lower(),
                "prompt_technique": technique,
                "question_number": i,
                "response": response
            }
            results.append(result)
            print(f"✅ {result['id']} gerada")
            time.sleep(1)

with open("aaaaquestoes_llms.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

print("✅ Arquivo JSON gerado: questoes_llms.json")