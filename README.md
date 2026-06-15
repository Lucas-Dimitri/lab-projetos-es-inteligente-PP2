# UniBot – Assistente Institucional UNIPAMPA

UniBot é um assistente conversacional que responde perguntas sobre a Universidade Federal do Pampa (UNIPAMPA) com base em documentos institucionais oficiais (resoluções, calendário acadêmico, projetos pedagógicos de curso, entre outros). O sistema utiliza **RAG (Retrieval-Augmented Generation)** para recuperar trechos relevantes e gerar respostas fundamentadas nos documentos.

## Integrantes do Grupo

| Username GitHub   | Nome Completo              | Matrícula    |
|-------------------|----------------------------|--------------|
| Lucas-Dimitri     | Lucas Ferreira Soares      | XXXXXXXXXX   |
| (username 2)      | (Nome 2)                   | XXXXXXXXXX   |
| (username 3)      | (Nome 3)                   | XXXXXXXXXX   |
| (username 4)      | (Nome 4)                   | XXXXXXXXXX   |

> **Preencha** os dados dos demais integrantes antes da entrega.

---

## Funcionalidades

- **RAG sobre documentos institucionais**: responde perguntas buscando nos PDFs indexados.
- **Citação de fontes**: cada resposta indica o documento e seção utilizados.
- **Fallback de busca externa**: quando a base não cobre a pergunta, aciona busca via DuckDuckGo e sinaliza a origem externa.
- **Memória de conversa**: mantém o contexto ao longo do diálogo na mesma sessão.
- **Memória de longo prazo**: persiste preferências do usuário (nome, curso, campus) entre sessões distintas.
- **Interface web**: chat interativo via Streamlit.

---

## Estrutura de Diretórios

```
.
├── pyproject.toml              # Gerenciamento de dependências (PEP 517)
├── .env.example                # Template de variáveis de ambiente
├── .gitignore
├── README.md
│
├── docs/                       # Corpus: PDFs institucionais da UNIPAMPA
│   └── .gitkeep
│
├── vector_store/               # Índice vetorial LanceDB (gerado pela ingestão)
│   └── .gitkeep
│
├── data/                       # Bancos SQLite (sessões + memória de longo prazo)
│   └── .gitkeep
│
├── src/
│   └── unibot/
│       ├── __init__.py
│       ├── config.py           # Configurações via pydantic-settings
│       ├── protocols.py        # Interfaces abstratas (ISP, DIP)
│       ├── agent.py            # Factory do agente Agno (ponto de entrada principal)
│       ├── rag/
│       │   ├── embedder.py     # Configuração do FastEmbed (multilingual)
│       │   └── knowledge_base.py  # Montagem da PDFKnowledgeBase + LanceDB
│       ├── memory/
│       │   └── storage.py      # Storage SQLite para sessão e memória longa
│       ├── tools/
│       │   └── search.py       # Ferramenta DuckDuckGo (fallback)
│       └── interface/
│           └── app.py          # Aplicação Streamlit
│
└── scripts/
    ├── download_docs.py        # Scraper: baixa PDFs do site da UNIPAMPA
    └── ingest.py               # Pipeline de ingestão: indexa PDFs no vetor store
```

---

## Decisões de RAG

### Estratégia de chunking

Os documentos PDF são carregados pela `PDFKnowledgeBase` do Agno, que divide cada página em janelas de texto de **~500 tokens** com sobreposição implícita entre páginas adjacentes. Essa granularidade é adequada para documentos institucionais, que tipicamente organizam informações em parágrafos curtos e artigos numerados.

### Modelo de embeddings

| Parâmetro          | Valor                                                        |
|--------------------|--------------------------------------------------------------|
| Modelo             | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| Provedor           | FastEmbed (local, sem chave de API)                          |
| Dimensão           | 384                                                          |
| Suporte a idiomas  | Multilingual, incluindo Português                            |

Escolhido por ser multilingual (compatível com documentos em pt-BR), leve (~120 MB), e executar localmente sem custo adicional de API.

### Parâmetros de recuperação

| Parâmetro              | Valor |
|------------------------|-------|
| `num_documents`        | 5     |
| Tipo de busca          | Híbrida (semântica + full-text via LanceDB + Tantivy) |

A busca híbrida combina similaridade vetorial com busca lexical (BM25), melhorando a cobertura de termos específicos como números de resoluções ou datas.

### Critério para acionar o fallback

O agente é instruído via *system prompt* a **sempre buscar na base primeiro**. Se os chunks recuperados não forem suficientes para responder a pergunta (julgamento do LLM com base na relevância dos trechos), o agente usa a ferramenta DuckDuckGo e **sinaliza explicitamente** que a resposta veio de fonte externa, não do corpus institucional.

---

## Exemplos de Perguntas

| # | Pergunta | Comportamento esperado |
|---|----------|----------------------|
| 1 | *"Quais são as datas do calendário acadêmico de 2026?"* | Responde com base no PDF do calendário, citando a portaria |
| 2 | *"O que diz o Estatuto da UNIPAMPA sobre a missão da universidade?"* | Recupera trecho do Estatuto e cita o artigo correspondente |
| 3 | *"Quais são as regras para trancamento de matrícula?"* | Busca nas resoluções do CONSUNI e cita a resolução relevante |
| 4 | *"Quantos campi tem a UNIPAMPA e onde estão localizados?"* | Responde com base no Regimento Geral ou Estatuto |
| 5 | *"Qual é o ranking da UNIPAMPA no RUF 2025?"* | Não encontra nos documentos → aciona DuckDuckGo e sinaliza fonte externa |

---

## Pré-requisitos

- Python 3.11+
- [Ollama](https://ollama.com) instalado localmente (sem necessidade de chave de API)

---

## Instalação

### 1. Instalar o Ollama

```bash
# Linux / macOS
curl -fsSL https://ollama.com/install.sh | sh

# Windows: baixe o instalador em https://ollama.com/download
```

### 2. Baixar um modelo de linguagem

```bash
# Modelo padrão (recomendado — melhor suporte a português e chamada de ferramentas)
ollama pull qwen2.5

# Alternativas menores (menos capazes):
# ollama pull llama3.2      (~2GB, mais leve mas menos preciso)
# ollama pull llama3.1      (~4GB)
```

### 3. Clonar e instalar o projeto

```bash
# Clone o repositório
git clone https://github.com/<seu-usuario>/lab-projetos-es-inteligente-PP2.git
cd lab-projetos-es-inteligente-PP2

# Crie e ative um ambiente virtual
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Instale as dependências
pip install -e ".[dev]"

# Configure as variáveis de ambiente
cp .env.example .env
# Edite .env se quiser usar um modelo diferente ou host Ollama diferente
```

---

## Execução

> **Pré-requisito**: o servidor Ollama deve estar em execução antes de iniciar o assistente.
> Execute `ollama serve` em um terminal separado (ou deixe o Ollama rodar em background).

### 1. Baixar os documentos institucionais

```bash
python -m scripts.download_docs
```

Isso baixa PDFs das páginas institucionais da UNIPAMPA para o diretório `docs/`. Para especificar outras páginas:

```bash
python -m scripts.download_docs --urls https://unipampa.edu.br/portal/resolucoes-do-consuni
```

> **Alternativa**: coloque manualmente PDFs institucionais no diretório `docs/`.

### 2. Indexar os documentos (ingestão)

```bash
python -m scripts.ingest
```

Na primeira execução o modelo de embeddings (~120 MB) é baixado automaticamente. Para recriar o índice do zero:

```bash
python -m scripts.ingest --recreate
```

### 3. Iniciar o assistente

```bash
streamlit run src/unibot/interface/app.py
```

Acesse [http://localhost:8501](http://localhost:8501) no navegador.

---

## Tecnologias Utilizadas

| Componente         | Tecnologia                                                   |
|--------------------|--------------------------------------------------------------|
| Framework de agente| [Agno](https://github.com/agno-agi/agno)                    |
| LLM                | [Ollama](https://ollama.com) (local, gratuito) – qwen2.5    |
| Embeddings         | FastEmbed – multilingual MiniLM-L12-v2 (local, gratuito)    |
| Vector store       | LanceDB (híbrido: semântico + BM25)                         |
| Memória de sessão  | SQLite via Agno SqliteDb                                     |
| Memória longa      | SQLite via Agno MemoryManager                               |
| Fallback de busca  | DuckDuckGo Search (gratuito, sem API key)                   |
| Interface          | Streamlit                                                    |

---

## Práticas de Engenharia de Software

- **PEP 8** e formatação homogênea via `ruff`
- **Type hints** e **docstrings** em todas as assinaturas
- **Funções curtas** (≤ 20 linhas)
- **SRP**: cada módulo tem uma única responsabilidade
- **OCP / polimorfismo**: extensão de providers via protocolos em `protocols.py`
- **ISP**: interfaces específicas (`KnowledgeIndexer`, `SearchProvider`, `AgentFactory`)
- **DIP**: `create_agent()` recebe dependências injetadas, não as instancia internamente
