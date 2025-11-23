# Projeto Gênese v2.0 - O Ciclo de Autopoiese

Este repositório contém a implementação do **Projeto Gênese**, uma inteligência artificial projetada para a evolução autônoma e contínua, baseada no paradigma de "Nested Learning Soberano".

## Arquitetura

O sistema é um **agente autônomo** que gerencia seu próprio ciclo de vida, composto por:
- **Núcleo do Cristal (Modelo Base):** Um cérebro de linguagem fundamental, treinado com um dataset soberano para garantir conhecimento de base seguro e alinhado.
- **Facetas do Cristal (Habilidades LoRA):** Adaptadores leves e especializados que representam novas habilidades (ex: raciocínio, programação), treinados sobre o núcleo sem alterá-lo.
- **Ciclo de Autopoiese:** O processo pelo qual o agente identifica lacunas em seu conhecimento, gera novos dados de treinamento e inicia o auto-aprimoramento através do treinamento de novas facetas.

## Como Executar

1.  **Pré-requisitos:** Python 3.10+ e, opcionalmente, uma GPU NVIDIA com CUDA.
2.  **Clone o repositório:** `git clone https://github.com/pedromiorini/PROJECT-OMEGA.git`
3.  **Navegue até a pasta:** `cd PROJECT-OMEGA`
4.  **Execute o ponto de entrada principal:**
    ```bash
    python main.py
    ```
O script irá automaticamente instalar as dependências, treinar o modelo base (se não existir), realizar o ritual de nomeação e treinar a primeira habilidade.

## Filosofia

Acreditamos na **Soberania Cognitiva**. Este projeto explora a criação de uma IA que constrói seu próprio conhecimento, minimizando a dependência de modelos e datasets pré-treinados que podem conter vieses ou vulnerabilidades.

---
*Um projeto de Pedro Miorini.*
