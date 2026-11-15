import os
import re
import sys
import json
import math
import time
import random
import logging
from pathlib import Path
from datetime import datetime
import requests
from pypdf import PdfReader
from dotenv import load_dotenv


# =========================================================
# 1. CONFIGURAÇÕES
# =========================================================
load_dotenv()

OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "https://ollama-api.ifba.edu.br"
).rstrip("/")

CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "qwen3:30b")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "qwen3-embedding:4b")

# Configurações de RAG / Seleção de Páginas por Instrumento
USAR_RAG_EMBEDDINGS = True
TOP_K_PAGINAS_POR_DOC = int(os.getenv("TOP_K_PAGINAS_POR_DOC", "12"))
MAX_CARACTERES_POR_PAGINA = int(os.getenv("MAX_CARACTERES_POR_PAGINA", "3500"))
EMBED_DIMENSIONS = None

# Configurações de Timeout e Janela de Contexto do Ollama
REQUEST_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "300"))
NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "32768"))

DOCUMENTOS = {
    "Plano Estratégico do Conselho Superior":
        Path("documentos/Plano Estrategico do Conselho Superior.pdf"),

    "Plano Estratégico do Tribunal":
        Path("documentos/Plano Estrategico do Tribunal.pdf"),

    "Cadeia de Valor do órgão":
        Path("documentos/cadeia de valor.pdf"),

    "Documento de auditoria":
        Path("documentos/Entendimento do objeto.pdf"),
}

DOCUMENTO_BASE_AUDITORIA = "Documento de auditoria"
DOC_ORGAO_SUPERIOR = "Plano Estratégico do Conselho Superior"
DOC_INSTITUICAO = "Plano Estratégico do Tribunal"
DOC_CADEIA_VALOR = "Cadeia de Valor do órgão"

DIRETORIO_SAIDA = Path("resultados")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logger = logging.getLogger(__name__)


# =========================================================
# 2. SCHEMAS JSON ESTRITOS (GRAMÁTICA GBNF DO OLLAMA)
# =========================================================
# O uso de JSON Schema diretamente no parâmetro "format" do Ollama
# impede que o modelo invente chaves como {"role": "system", "content": "..."}
# e garante que ele preencha estritamente os campos da auditoria.

SCHEMA_ENTENDIMENTO_AUDITORIA = {
    "type": "object",
    "properties": {
        "objeto_da_auditoria": {"type": "string"},
        "objetivo_da_auditoria": {"type": "string"},
        "escopo_e_entregas_principais": {
            "type": "array",
            "items": {"type": "string"}
        },
        "termos_chave_para_busca": {
            "type": "array",
            "items": {"type": "string"}
        },
        "paginas_fonte": {
            "type": "array",
            "items": {"type": "integer"}
        }
    },
    "required": [
        "objeto_da_auditoria",
        "objetivo_da_auditoria",
        "escopo_e_entregas_principais",
        "termos_chave_para_busca",
        "paginas_fonte"
    ]
}

SCHEMA_ALINHAMENTO_ESTRATEGICO = {
    "type": "object",
    "properties": {
        "instrumento_analisado": {"type": "string"},
        "declaracao_ausencia": {
            "type": ["string", "null"]
        },
        "objetivos_identificados": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "objetivo_estrategico": {"type": "string"},
                    "classificacao_correlacao": {
                        "type": "string",
                        "enum": [
                            "Correlação Direta Comprovada",
                            "Correlação Parcial",
                            "Ausência de Correlação Documental"
                        ]
                    },
                    "indicadores": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "nome_indicador": {"type": "string"},
                                "formula_ou_metrica": {"type": "string"},
                                "meta": {"type": "string"},
                                "periodo": {"type": "string"}
                            },
                            "required": [
                                "nome_indicador",
                                "formula_ou_metrica",
                                "meta",
                                "periodo"
                            ]
                        }
                    },
                    "evidencias_documentais": {
                        "type": "object",
                        "properties": {
                            "documento_origem": {"type": "string"},
                            "localizacao": {"type": "string"},
                            "evidencia_literal": {"type": "string"},
                            "justificativa": {"type": "string"}
                        },
                        "required": [
                            "documento_origem",
                            "localizacao",
                            "evidencia_literal",
                            "justificativa"
                        ]
                    }
                },
                "required": [
                    "objetivo_estrategico",
                    "classificacao_correlacao",
                    "indicadores",
                    "evidencias_documentais"
                ]
            }
        },
        "lacunas_identificadas_no_instrumento": {
            "type": "array",
            "items": {"type": "string"}
        }
    },
    "required": [
        "instrumento_analisado",
        "declaracao_ausencia",
        "objetivos_identificados",
        "lacunas_identificadas_no_instrumento"
    ]
}

SCHEMA_CADEIA_VALOR = {
    "type": "object",
    "properties": {
        "instrumento_analisado": {"type": "string"},
        "declaracao_ausencia": {
            "type": ["string", "null"]
        },
        "macroprocessos_identificados": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "macroprocesso": {"type": "string"},
                    "tipo_ou_nivel_macroprocesso": {"type": "string"},
                    "classificacao_correlacao": {
                        "type": "string",
                        "enum": [
                            "Correlação Direta Comprovada",
                            "Correlação Parcial",
                            "Ausência de Correlação Documental"
                        ]
                    },
                    "evidencias_documentais": {
                        "type": "object",
                        "properties": {
                            "documento_origem": {"type": "string"},
                            "localizacao": {"type": "string"},
                            "evidencia_literal": {"type": "string"},
                            "justificativa": {"type": "string"}
                        },
                        "required": [
                            "documento_origem",
                            "localizacao",
                            "evidencia_literal",
                            "justificativa"
                        ]
                    }
                },
                "required": [
                    "macroprocesso",
                    "tipo_ou_nivel_macroprocesso",
                    "classificacao_correlacao",
                    "evidencias_documentais"
                ]
            }
        },
        "lacunas_identificadas_no_instrumento": {
            "type": "array",
            "items": {"type": "string"}
        }
    },
    "required": [
        "instrumento_analisado",
        "declaracao_ausencia",
        "macroprocessos_identificados",
        "lacunas_identificadas_no_instrumento"
    ]
}

SCHEMA_SINTESE_EXECUTIVA = {
    "type": "object",
    "properties": {
        "resumo_geral": {"type": "string"},
        "alinhamentos_comprovados": {
            "type": "array",
            "items": {"type": "string"}
        },
        "alinhamentos_nao_identificados": {
            "type": "array",
            "items": {"type": "string"}
        }
    },
    "required": [
        "resumo_geral",
        "alinhamentos_comprovados",
        "alinhamentos_nao_identificados"
    ]
}


# =========================================================
# 3. INSTRUÇÕES DO SISTEMA
# =========================================================

SYSTEM_INSTRUCTION = """/no_think
Você é um auditor especialista em auditoria governamental, planejamento estratégico institucional e governança pública.
Sua única função é extrair correlações documentais estritas em formato JSON de acordo com o schema exigido.
NUNCA elabore análise SWOT, NUNCA assuma papel de consultor genérico e NUNCA gere chaves como "role" ou "content".
Baseie-se exclusivamente nas evidências expressas nas páginas fornecidas.
"""


# =========================================================
# 4. VERIFICAÇÃO E PRÉ-AQUECIMENTO DO SERVIDOR OLLAMA
# =========================================================

def verificar_servidor_ollama(session):
    url = f"{OLLAMA_BASE_URL}/api/tags"
    logger.info("Verificando modelos disponíveis em: %s", url)

    try:
        resposta = session.get(url, timeout=30)
        resposta.raise_for_status()
        dados = resposta.json()

        modelos_disponiveis = [
            m.get("name") for m in dados.get("models", [])
        ]

        logger.info(
            "Servidor Ollama acessível. Modelos disponíveis (%d): %s",
            len(modelos_disponiveis),
            ", ".join(modelos_disponiveis)
        )

        if CHAT_MODEL not in modelos_disponiveis:
            logger.warning(
                "Modelo de chat '%s' não encontrado em /api/tags.",
                CHAT_MODEL
            )

        if USAR_RAG_EMBEDDINGS and EMBED_MODEL not in modelos_disponiveis:
            logger.warning(
                "Modelo de embedding '%s' não encontrado em /api/tags.",
                EMBED_MODEL
            )

        return modelos_disponiveis

    except requests.exceptions.RequestException as erro:
        logger.error("Falha ao conectar ao servidor Ollama do IFBA: %s", erro)
        raise


def pre_aquecer_modelo_chat(session):
    url = f"{OLLAMA_BASE_URL}/api/chat"
    logger.info("Pré-aquecendo o modelo '%s' na GPU do servidor...", CHAT_MODEL)

    payload = {
        "model": CHAT_MODEL,
        "messages": [{"role": "user", "content": "/no_think Responda apenas ok"}],
        "stream": False,
        "think": False,
        "options": {
            "num_predict": 2,
            "num_ctx": NUM_CTX
        }
    }

    try:
        resposta = session.post(url, json=payload, timeout=90)
        if resposta.ok:
            logger.info("Modelo '%s' carregado e pronto na GPU.", CHAT_MODEL)
        else:
            logger.warning(
                "Aviso no pré-aquecimento (HTTP %s): prosseguindo.",
                resposta.status_code
            )
    except requests.exceptions.RequestException as erro:
        logger.warning("Pré-aquecimento falhou (%s). Prosseguindo...", erro)


# =========================================================
# 5. VALIDAÇÃO E EXTRAÇÃO DOS PDFs
# =========================================================

def validar_documentos(documentos):
    arquivos_validos = {}

    for nome, caminho in documentos.items():
        if not caminho.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {caminho}")
        if not caminho.is_file():
            raise ValueError(f"O caminho não representa um arquivo: {caminho}")
        if caminho.suffix.lower() != ".pdf":
            raise ValueError(f"Formato inválido para {nome}. Esperado PDF.")
        if caminho.stat().st_size == 0:
            raise ValueError(f"Arquivo vazio: {caminho}")

        arquivos_validos[nome] = caminho
        logger.info("Documento validado: %s", nome)

    return arquivos_validos


def limpar_texto_pagina(texto):
    texto = re.sub(r"[ \t]+", " ", texto)
    texto = re.sub(r"\n{3,}", "\n\n", texto)
    texto = texto.strip()
    if len(texto) > MAX_CARACTERES_POR_PAGINA:
        texto = texto[:MAX_CARACTERES_POR_PAGINA] + "... [truncado]"
    return texto


def extrair_paginas_pdf(documentos):
    documentos_extraidos = {}

    for nome, caminho in documentos.items():
        logger.info("Extraindo texto do PDF: %s", nome)
        paginas = []

        leitor = PdfReader(str(caminho))
        for indice, pagina in enumerate(leitor.pages, start=1):
            texto_bruto = (pagina.extract_text() or "").strip()
            texto = limpar_texto_pagina(texto_bruto)
            if texto:
                paginas.append({
                    "documento": nome,
                    "pagina": indice,
                    "texto": texto
                })

        if not paginas:
            raise ValueError(f"Nenhum texto extraível encontrado no PDF: {nome}")

        documentos_extraidos[nome] = paginas
        logger.info(
            "Extração concluída: %s (%d páginas com texto)",
            nome,
            len(paginas)
        )

    return documentos_extraidos


# =========================================================
# 6. EMBEDDINGS (/api/embed) E SELEÇÃO SEMÂNTICA
# =========================================================

def gerar_embeddings_com_retry(session, textos, max_tentativas=5):
    url = f"{OLLAMA_BASE_URL}/api/embed"
    codigos_retentaveis = {429, 500, 502, 503, 504}

    payload = {
        "model": EMBED_MODEL,
        "input": textos
    }
    if EMBED_DIMENSIONS is not None:
        payload["dimensions"] = EMBED_DIMENSIONS

    for tentativa in range(1, max_tentativas + 1):
        try:
            resposta = session.post(url, json=payload, timeout=REQUEST_TIMEOUT)
            if resposta.status_code in codigos_retentaveis:
                raise requests.exceptions.HTTPError(
                    f"HTTP {resposta.status_code}: {resposta.text}",
                    response=resposta
                )

            resposta.raise_for_status()
            dados = resposta.json()
            embeddings = dados.get("embeddings")
            if not embeddings:
                raise RuntimeError("A API /api/embed não retornou 'embeddings'.")
            return embeddings

        except requests.exceptions.RequestException as erro:
            if tentativa == max_tentativas:
                raise
            espera = min(2 ** tentativa, 32) + random.uniform(0, 2)
            logger.warning(
                "Falha temporária em /api/embed. Tentativa %d/%d. Aguardando %.1fs...",
                tentativa, max_tentativas, espera
            )
            time.sleep(espera)


def similaridade_cosseno(vetor_a, vetor_b):
    produto_escalar = sum(a * b for a, b in zip(vetor_a, vetor_b))
    norma_a = math.sqrt(sum(a * a for a in vetor_a))
    norma_b = math.sqrt(sum(b * b for b in vetor_b))
    if norma_a == 0.0 or norma_b == 0.0:
        return 0.0
    return produto_escalar / (norma_a * norma_b)


def calcular_bonus_palavras_chave(texto_pagina, tipo_documento):
    """
    Adiciona um pequeno bônus ao score de similaridade para páginas que
    contenham termos estruturais de metas, indicadores, objetivos ou macroprocessos,
    garantindo que quadros de metas e cadeias de valor não fiquem de fora.
    """
    texto_lower = texto_pagina.lower()
    bonus = 0.0

    if tipo_documento == "estrategico":
        termos = ["objetivo estratégico", "indicador", "meta", "fórmula", "periodicidade", "índice"]
    else:
        termos = ["macroprocesso", "cadeia de valor", "finalístico", "governança", "suporte", "gestão"]

    for termo in termos:
        if termo in texto_lower:
            bonus += 0.025

    return min(bonus, 0.12)


def selecionar_paginas_documento(session, nome_doc, paginas, texto_consulta_referencia, tipo_doc="estrategico"):
    if not USAR_RAG_EMBEDDINGS or len(paginas) <= TOP_K_PAGINAS_POR_DOC:
        return paginas

    logger.info(
        "Ranqueando as %d páginas de '%s' via embeddings (%s)...",
        len(paginas),
        nome_doc,
        EMBED_MODEL
    )

    vetor_ref = gerar_embeddings_com_retry(session, [texto_consulta_referencia])[0]
    vetores_paginas = gerar_embeddings_com_retry(session, [p["texto"] for p in paginas])

    paginas_pontuadas = []
    for pagina, vetor in zip(paginas, vetores_paginas):
        sim_base = similaridade_cosseno(vetor_ref, vetor)
        bonus = calcular_bonus_palavras_chave(pagina["texto"], tipo_doc)
        score_final = sim_base + bonus

        copia = dict(pagina)
        copia["score_similaridade"] = round(sim_base, 4)
        copia["score_final"] = round(score_final, 4)
        paginas_pontuadas.append(copia)

    melhores = sorted(
        paginas_pontuadas,
        key=lambda x: x["score_final"],
        reverse=True
    )[:TOP_K_PAGINAS_POR_DOC]

    melhores.sort(key=lambda x: x["pagina"])

    logger.info(
        "Páginas selecionadas de '%s': %s",
        nome_doc,
        [p["pagina"] for p in melhores]
    )
    return melhores


# =========================================================
# 7. CHAMADA AO OLLAMA COM STREAMING E SCHEMA VALIDADO
# =========================================================

def extrair_json_limpo(texto_resposta):
    texto_sem_think = re.sub(
        r"<think>.*?</think>",
        "",
        texto_resposta,
        flags=re.DOTALL
    ).strip()

    match_bloco = re.search(
        r"```(?:json)?\s*(\{.*\})\s*```",
        texto_sem_think,
        flags=re.DOTALL
    )
    if match_bloco:
        texto_sem_think = match_bloco.group(1).strip()
    else:
        inicio = texto_sem_think.find("{")
        fim = texto_sem_think.rfind("}")
        if inicio != -1 and fim != -1 and fim > inicio:
            texto_sem_think = texto_sem_think[inicio:fim + 1]

    return json.loads(texto_sem_think)


def chamar_ollama_json(session, prompt_usuario, json_schema, descricao_etapa, max_tentativas=4):
    """
    Executa chamada em modo STREAM com JSON Schema estrito (GBNF) e verifica
    se todas as chaves obrigatórias do schema estão presentes na resposta.
    As instruções ficam no final do prompt para evitar perda por truncamento de contexto.
    """
    url = f"{OLLAMA_BASE_URL}/api/chat"
    codigos_retentaveis = {429, 500, 502, 503, 504}
    chaves_obrigatorias = set(json_schema.get("required", []))

    mensagens = [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": prompt_usuario}
    ]

    payload = {
        "model": CHAT_MODEL,
        "messages": mensagens,
        "stream": True,
        "think": False,
        "format": json_schema,
        "options": {
            "temperature": 0.0,
            "num_ctx": NUM_CTX
        }
    }

    for tentativa in range(1, max_tentativas + 1):
        try:
            logger.info(
                "[%s] Enviando requisição ao Ollama (%d caracteres | tentativa %d/%d)...",
                descricao_etapa,
                len(prompt_usuario),
                tentativa,
                max_tentativas
            )

            inicio_req = time.monotonic()
            partes_conteudo = []
            primeiro_token = False

            with session.post(
                url,
                json=payload,
                stream=True,
                timeout=(30, REQUEST_TIMEOUT)
            ) as resposta:

                if resposta.status_code in codigos_retentaveis:
                    raise requests.exceptions.HTTPError(
                        f"HTTP {resposta.status_code}: {resposta.text}",
                        response=resposta
                    )

                resposta.raise_for_status()

                for linha in resposta.iter_lines(decode_unicode=True):
                    if not linha:
                        continue

                    pacote = json.loads(linha)
                    if "error" in pacote:
                        raise RuntimeError(f"Erro na stream do Ollama: {pacote['error']}")

                    trecho = pacote.get("message", {}).get("content", "")
                    if trecho:
                        if not primeiro_token:
                            primeiro_token = True
                            logger.info(
                                "[%s] Primeiro token recebido em %.1fs.",
                                descricao_etapa,
                                time.monotonic() - inicio_req
                            )
                        partes_conteudo.append(trecho)

                    if pacote.get("done") is True:
                        break

            conteudo_bruto = "".join(partes_conteudo).strip()
            if not conteudo_bruto:
                raise RuntimeError("Resposta vazia retornada pelo Ollama.")

            dados_json = extrair_json_limpo(conteudo_bruto)

            # Verifica se o modelo gerou as chaves do schema e não {"role": "system", ...}
            if not isinstance(dados_json, dict) or not chaves_obrigatorias.issubset(dados_json.keys()):
                raise ValueError(
                    f"JSON fora do schema esperado. Chaves recebidas: {list(dados_json.keys())}"
                )

            logger.info(
                "[%s] Concluído com sucesso em %.1fs.",
                descricao_etapa,
                time.monotonic() - inicio_req
            )
            return dados_json

        except (requests.exceptions.RequestException, json.JSONDecodeError, RuntimeError, ValueError) as erro:
            if tentativa == max_tentativas:
                logger.error("[%s] Falha definitiva após %d tentativas: %s", descricao_etapa, max_tentativas, erro)
                raise

            espera = min(2 ** tentativa, 20) + random.uniform(0, 2)
            logger.warning(
                "[%s] Tentativa %d/%d falhou (%s). Retentando em %.1fs...",
                descricao_etapa,
                tentativa,
                max_tentativas,
                erro,
                espera
            )
            time.sleep(espera)


# =========================================================
# 8. ETAPAS DE ANÁLISE POR INSTRUMENTO
# =========================================================

def formatar_paginas_para_prompt(nome_doc, paginas):
    blocos = []
    for p in paginas:
        blocos.append(
            f"--- [Documento: {nome_doc} | Página: {p['pagina']}] ---\n{p['texto']}"
        )
    return "\n\n".join(blocos)


def extrair_entendimento_auditoria(session, paginas_auditoria):
    """
    Etapa 1: Analisa exclusivamente o Documento de Auditoria para extrair o Objeto,
    o Objetivo da Auditoria, o Escopo/Entregas e os termos-chave.
    """
    paginas_selecionadas = paginas_auditoria[:14]
    texto_doc = formatar_paginas_para_prompt(DOCUMENTO_BASE_AUDITORIA, paginas_selecionadas)

    prompt = f"""/no_think
DOCUMENTO FONTE ({DOCUMENTO_BASE_AUDITORIA}):

{texto_doc}

=========================================================
TAREFA OBRIGATÓRIA (ETAPA 1 - ENTENDIMENTO DO OBJETO DA AUDITORIA):
Com base EXCLUSIVAMENTE nas páginas acima do "{DOCUMENTO_BASE_AUDITORIA}", identifique e extraia em JSON:
1. "objeto_da_auditoria": Descrição clara e completa do objeto da auditoria/fiscalização.
2. "objetivo_da_auditoria": Qual é a finalidade/objetivo central da auditoria.
3. "escopo_e_entregas_principais": Lista dos principais tópicos, entregas, processos ou áreas avaliadas na auditoria.
4. "termos_chave_para_busca": Lista de 8 a 15 termos técnicos ou temas centrais do objeto da auditoria para buscar alinhamento nos planos estratégicos e na cadeia de valor.
5. "paginas_fonte": Lista dos números de página de onde essas informações foram extraídas.
"""
    return chamar_ollama_json(
        session,
        prompt,
        SCHEMA_ENTENDIMENTO_AUDITORIA,
        descricao_etapa="1/5 Entendimento da Auditoria"
    )


def analisar_plano_estrategico(session, nome_instrumento, paginas_selecionadas, entendimento_auditoria):
    """
    Etapa 2 e 3: Analisa um Plano Estratégico específico (Conselho Superior ou Tribunal)
    contra o Objeto/Objetivo da Auditoria, extraindo Objetivos Estratégicos, Indicadores e Metas.
    """
    texto_doc = formatar_paginas_para_prompt(nome_instrumento, paginas_selecionadas)
    resumo_auditoria = json.dumps(entendimento_auditoria, ensure_ascii=False, indent=2)

    prompt = f"""/no_think
FONTES DOCUMENTAIS DO INSTRUMENTO "{nome_instrumento}":

{texto_doc}

=========================================================
PARÂMETRO DE AUDITORIA (EXTRAÍDO DO DOCUMENTO DE AUDITORIA):
{resumo_auditoria}

=========================================================
TAREFA OBRIGATÓRIA DE AUDITORIA:
Analise as páginas do instrumento "{nome_instrumento}" fornecidas acima e identifique quais **Objetivos Estratégicos**, **Indicadores** e **Metas** estão alinhados ao Objeto e ao Objetivo da Auditoria.

REGRAS ESTRITAS:
1. Classifique cada objetivo identificado estritamente em uma das opções:
   - "Correlação Direta Comprovada": menciona explicitamente o objeto da auditoria, suas entregas ou a finalidade exata da fiscalização.
   - "Correlação Parcial": o objeto da auditoria é meio, insumo ou subtema necessário para atingir o objetivo estratégico, com evidência documental.
   - "Ausência de Correlação Documental": sem evidência suficiente.
2. Para cada Objetivo Estratégico alinhado, extraia TODOS os seus **Indicadores** e respectivas **Metas** (transcrevendo exatamente nome_indicador, formula_ou_metrica, meta e periodo como constam no documento; se algum dado não constar nas páginas, informe "Não especificado nas páginas analisadas").
3. Em "evidencias_documentais", informe:
   - "documento_origem": "{nome_instrumento}"
   - "localizacao": Página exata, seção ou código do objetivo.
   - "evidencia_literal": Trecho literal extraído do texto que comprova o vínculo.
   - "justificativa": Explicação objetiva de como este objetivo estratégico se relaciona com o objeto/objetivo da auditoria.
4. Caso nenhum objetivo estratégico tenha correlação comprovada, preencha "declaracao_ausencia" com: "Não foi identificada correlação documental direta nas fontes disponibilizadas." (caso contrário, use null).
"""
    return chamar_ollama_json(
        session,
        prompt,
        SCHEMA_ALINHAMENTO_ESTRATEGICO,
        descricao_etapa=f"Análise: {nome_instrumento}"
    )


def analisar_cadeia_valor(session, nome_instrumento, paginas_selecionadas, entendimento_auditoria):
    """
    Etapa 4: Analisa a Cadeia de Valor do órgão contra o Objeto/Objetivo da Auditoria,
    identificando os Macroprocessos relacionados.
    """
    texto_doc = formatar_paginas_para_prompt(nome_instrumento, paginas_selecionadas)
    resumo_auditoria = json.dumps(entendimento_auditoria, ensure_ascii=False, indent=2)

    prompt = f"""/no_think
FONTES DOCUMENTAIS DO INSTRUMENTO "{nome_instrumento}":

{texto_doc}

=========================================================
PARÂMETRO DE AUDITORIA (EXTRAÍDO DO DOCUMENTO DE AUDITORIA):
{resumo_auditoria}

=========================================================
TAREFA OBRIGATÓRIA DE AUDITORIA (CADEIA DE VALOR):
Analise as páginas do documento "{nome_instrumento}" fornecidas acima e identifique quais **Macroprocessos** (e processos/subprocessos vinculados) da Cadeia de Valor do órgão possuem relação documental com o Objeto e o Objetivo da Auditoria.

REGRAS ESTRITAS:
1. Não invente macroprocessos; transcreva o nome exato conforme consta na Cadeia de Valor.
2. Informe em "tipo_ou_nivel_macroprocesso" a categoria na Cadeia de Valor (ex.: Finalístico, Governança/Estratégico, Suporte/Apoio/Gestão).
3. Classifique o vínculo estritamente em:
   - "Correlação Direta Comprovada"
   - "Correlação Parcial"
   - "Ausência de Correlação Documental"
4. Apresente obrigatoriamente em "evidencias_documentais":
   - "documento_origem": "{nome_instrumento}"
   - "localizacao": Página, seção ou item.
   - "evidencia_literal": Trecho literal extraído do documento.
   - "justificativa": Explicação objetiva da relação do macroprocesso com o objeto/objetivo da auditoria.
5. Se não houver correlação comprovada, preencha "declaracao_ausencia" com: "Não foi identificada correlação documental direta nas fontes disponibilizadas." (caso contrário, use null).
"""
    return chamar_ollama_json(
        session,
        prompt,
        SCHEMA_CADEIA_VALOR,
        descricao_etapa=f"Análise: {nome_instrumento}"
    )


def gerar_sintese_executiva(session, entendimento, res_superior, res_instituicao, res_cadeia):
    """
    Etapa 5: Gera a síntese executiva consolidando os alinhamentos comprovados e não identificados.
    """
    dados_consolidados = {
        "entendimento_auditoria": entendimento,
        "orgao_superior_objetivos": [
            {
                "objetivo": o["objetivo_estrategico"],
                "classificacao": o["classificacao_correlacao"],
                "qtd_indicadores": len(o.get("indicadores", []))
            }
            for o in res_superior.get("objetivos_identificados", [])
        ],
        "instituicao_objetivos": [
            {
                "objetivo": o["objetivo_estrategico"],
                "classificacao": o["classificacao_correlacao"],
                "qtd_indicadores": len(o.get("indicadores", []))
            }
            for o in res_instituicao.get("objetivos_identificados", [])
        ],
        "cadeia_valor_macroprocessos": [
            {
                "macroprocesso": m["macroprocesso"],
                "classificacao": m["classificacao_correlacao"]
            }
            for m in res_cadeia.get("macroprocessos_identificados", [])
        ]
    }

    prompt = f"""/no_think
RESULTADOS EXTRAÍDOS DA AUDITORIA:
{json.dumps(dados_consolidados, ensure_ascii=False, indent=2)}

=========================================================
TAREFA OBRIGATÓRIA (SÍNTESE EXECUTIVA):
Elabore a Síntese Executiva em JSON indicando:
- "resumo_geral": Síntese objetiva do alinhamento entre o objeto da auditoria, os planos estratégicos (Órgão Superior e Instituição) e os macroprocessos da Cadeia de Valor.
- "alinhamentos_comprovados": Lista objetiva dos objetivos estratégicos, indicadores/metas e macroprocessos cujo alinhamento foi documentalmente comprovado.
- "alinhamentos_nao_identificados": Lista de ausências de correlação ou lacunas documentais observadas.
"""
    return chamar_ollama_json(
        session,
        prompt,
        SCHEMA_SINTESE_EXECUTIVA,
        descricao_etapa="5/5 Síntese Executiva"
    )


# =========================================================
# 9. SALVAR RESULTADO EM JSON
# =========================================================

def salvar_resultado(dados_json):
    DIRETORIO_SAIDA.mkdir(parents=True, exist_ok=True)
    data = datetime.now().strftime("%Y%m%d_%H%M%S")
    caminho = DIRETORIO_SAIDA / f"analise_auditoria_{data}.json"

    texto_json = json.dumps(dados_json, ensure_ascii=False, indent=2)
    caminho.write_text(texto_json, encoding="utf-8")
    logger.info("Relatório JSON salvo em: %s", caminho)
    return caminho


# =========================================================
# 10. EXECUÇÃO PRINCIPAL
# =========================================================

def main():
    session = requests.Session()
    session.headers.update({"Content-Type": "application/json"})

    try:
        logger.info("Iniciando análise de alinhamento estratégico e Cadeia de Valor via Ollama IFBA.")

        verificar_servidor_ollama(session)
        documentos_validos = validar_documentos(DOCUMENTOS)
        documentos_extraidos = extrair_paginas_pdf(documentos_validos)

        pre_aquecer_modelo_chat(session)

        # 1. Extrai o Objeto e Objetivo da Auditoria do documento base
        paginas_auditoria = documentos_extraidos[DOCUMENTO_BASE_AUDITORIA]
        entendimento_auditoria = extrair_entendimento_auditoria(
            session,
            paginas_auditoria
        )

        # Monta consulta semântica enriquecida para buscar nos Planos e na Cadeia de Valor
        consulta_semantica_planos = (
            f"Objeto da auditoria: {entendimento_auditoria['objeto_da_auditoria']}\n"
            f"Objetivo da auditoria: {entendimento_auditoria['objetivo_da_auditoria']}\n"
            f"Termos-chave: {', '.join(entendimento_auditoria.get('termos_chave_para_busca', []))}\n"
            f"Objetivos estratégicos, indicadores, metas, fórmulas e métricas."
        )

        consulta_semantica_cadeia = (
            f"Objeto da auditoria: {entendimento_auditoria['objeto_da_auditoria']}\n"
            f"Objetivo da auditoria: {entendimento_auditoria['objetivo_da_auditoria']}\n"
            f"Termos-chave: {', '.join(entendimento_auditoria.get('termos_chave_para_busca', []))}\n"
            f"Cadeia de valor, macroprocessos finalísticos, governança, gestão e suporte."
        )

        # 2. Seleciona páginas relevantes e analisa o Plano Estratégico do Conselho Superior
        paginas_conselho = selecionar_paginas_documento(
            session,
            DOC_ORGAO_SUPERIOR,
            documentos_extraidos[DOC_ORGAO_SUPERIOR],
            consulta_semantica_planos,
            tipo_doc="estrategico"
        )
        res_orgao_superior = analisar_plano_estrategico(
            session,
            DOC_ORGAO_SUPERIOR,
            paginas_conselho,
            entendimento_auditoria
        )

        # 3. Seleciona páginas relevantes e analisa o Plano Estratégico do Tribunal
        paginas_tribunal = selecionar_paginas_documento(
            session,
            DOC_INSTITUICAO,
            documentos_extraidos[DOC_INSTITUICAO],
            consulta_semantica_planos,
            tipo_doc="estrategico"
        )
        res_instituicao = analisar_plano_estrategico(
            session,
            DOC_INSTITUICAO,
            paginas_tribunal,
            entendimento_auditoria
        )

        # 4. Seleciona páginas relevantes e analisa a Cadeia de Valor do órgão
        paginas_cadeia = selecionar_paginas_documento(
            session,
            DOC_CADEIA_VALOR,
            documentos_extraidos[DOC_CADEIA_VALOR],
            consulta_semantica_cadeia,
            tipo_doc="cadeia_valor"
        )
        res_cadeia_valor = analisar_cadeia_valor(
            session,
            DOC_CADEIA_VALOR,
            paginas_cadeia,
            entendimento_auditoria
        )

        # 5. Consolida Lacunas e Síntese Executiva
        lacunas_consolidadas = []
        for nome_inst, resultado_inst in [
            (DOC_ORGAO_SUPERIOR, res_orgao_superior),
            (DOC_INSTITUICAO, res_instituicao),
            (DOC_CADEIA_VALOR, res_cadeia_valor)
        ]:
            for lacuna in resultado_inst.get("lacunas_identificadas_no_instrumento", []):
                lacunas_consolidadas.append({
                    "instrumento_ou_item": nome_inst,
                    "descricao_lacuna": lacuna
                })

        sintese_executiva = gerar_sintese_executiva(
            session,
            entendimento_auditoria,
            res_orgao_superior,
            res_instituicao,
            res_cadeia_valor
        )

        # Monta o JSON final estruturado
        resultado_final = {
            "entendimento_da_auditoria": entendimento_auditoria,
            "alinhamento_orgao_superior": {
                "instrumento": DOC_ORGAO_SUPERIOR,
                "declaracao_ausencia": res_orgao_superior.get("declaracao_ausencia"),
                "objetivos_identificados": res_orgao_superior.get("objetivos_identificados", [])
            },
            "alinhamento_instituicao": {
                "instrumento": DOC_INSTITUICAO,
                "declaracao_ausencia": res_instituicao.get("declaracao_ausencia"),
                "objetivos_identificados": res_instituicao.get("objetivos_identificados", [])
            },
            "macroprocessos_cadeia_valor": {
                "instrumento": DOC_CADEIA_VALOR,
                "declaracao_ausencia": res_cadeia_valor.get("declaracao_ausencia"),
                "macroprocessos_identificados": res_cadeia_valor.get("macroprocessos_identificados", [])
            },
            "lacunas_e_ausencias": lacunas_consolidadas,
            "sintese_executiva": sintese_executiva
        }

        caminho = salvar_resultado(resultado_final)

        print("\n" + "=" * 70)
        print("[Objetivos Estratégicos, Indicadores, Metas e Macroprocessos - JSON]")
        print("=" * 70)
        print(json.dumps(resultado_final, ensure_ascii=False, indent=2))
        print("\nRelatório salvo em:", caminho)

    except FileNotFoundError as erro:
        logger.critical("Arquivo não encontrado: %s", erro)
        sys.exit(1)

    except requests.exceptions.RequestException as erro:
        logger.critical(
            "Falha de comunicação com o servidor Ollama do IFBA (%s): %s",
            OLLAMA_BASE_URL,
            erro
        )
        sys.exit(1)

    except Exception as erro:
        logger.exception("Falha inesperada na execução: %s", erro)
        sys.exit(1)

    finally:
        session.close()


if __name__ == "__main__":
    main()