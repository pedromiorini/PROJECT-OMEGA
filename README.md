
# PROJECT OMEGA (Gênese)

## Visão Geral

Este repositório representa o legado do **Projeto Gênese**, uma jornada para criar um agente de IA autônomo capaz de aprender, raciocinar e executar tarefas de programação. Este código, versão **v12.2**, implementa a arquitetura **GTR (Generate, Test, Refine)**, a mais bem-sucedida e estável que desenvolvemos, agora com **Autonomia de Ambiente (Escrita de Arquivos e Git)**.

## A Jornada

O projeto evoluiu através de várias versões, cada uma ensinando uma lição valiosa:
- **v1.0 - v8.0:** Luta inicial com bugs, setup e a exploração da evolução de pesos (EGGROLL). Concluímos que, embora fascinante, era a ferramenta errada para ensinar conceitos de programação do zero. O sinal de fitness era muito fraco e o espaço de busca, infinito.
- **v9.0:** A introdução do "Cérebro Especialista" (`deepseek-math-7b-instruct`), que nos deu a matéria-prima correta para o raciocínio.
- **v10.0 (Este código):** A grande pivotada. Abandonamos a evolução de pesos e implementamos o ciclo GTR. O Gênese aprendeu a gerar código, testá-lo, analisar o erro e se autocorrigir, simulando o fluxo de trabalho de um programador real. **Esta foi a nossa maior vitória.**
- **v10.0.1 (Atualização de Saúde):** Implementação de mitigação de riscos de hardware, separação de dependências e melhoria na resiliência do ciclo GTR.
- **v11.0 - v12.0:** Tentativas de dar ao Gênese autonomia sobre seu ambiente (escrever arquivos, usar Git).
- **v12.2 (Atualização de Autonomia):** Consolidação das ferramentas de autonomia de ambiente (`escrever_arquivo` e `executar_git`) na classe `FerramentasSeguras`, permitindo que o Gênese interaja com o sistema de arquivos e o controle de versão.

## Arquitetura v10.0: GTR (Generate, Test, Refine)

O `main.py` neste repositório implementa um ciclo de aprendizado de habilidades em três fases:

1.  **Geração:** O Cérebro Especialista recebe uma tarefa (ex: "Crie a função X") e gera uma primeira versão do código.
2.  **Teste:** O sistema executa o código gerado contra uma unidade de teste.
3.  **Refinamento:** Se o teste falhar, o sistema captura o `Traceback` do erro e o alimenta de volta ao Cérebro, instruindo-o a analisar o erro e gerar uma versão corrigida. O ciclo se repete até o sucesso.

Esta arquitetura provou ser robusta, resiliente e a base para qualquer desenvolvimento futuro do Projeto Gênese.

## Como Executar

**1. Instalação de Dependências (Mitigação 2)**

Crie um ambiente virtual e instale as dependências usando o arquivo `requirements.txt`:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**2. Execução do Script**

Execute o script principal:

```bash
python main.py
```

**3. Configuração Opcional (Mitigação 4)**

Para alterar o *timeout* de execução do código gerado (padrão é 15 segundos), defina a variável de ambiente `GENESIS_TIMEOUT_SECS`:

```bash
export GENESIS_TIMEOUT_SECS=30
python main.py
```

**4. Requisitos de Hardware (Mitigação 1)**

O modelo padrão (`deepseek-math-7b-instruct`) é grande e requer uma **GPU com pelo menos 8GB de VRAM** para um desempenho ideal. Se nenhuma GPU for detectada, o script tentará carregar o modelo na CPU, o que será significativamente mais lento.

---
*Este repositório foi atualizado autonomamente como parte da missão v12.2, orquestrada por Manus e Pedro Miorini.*

## Ferramentas de Autonomia (v12.2)

A classe `FerramentasSeguras` foi estendida para incluir as seguintes capacidades, que permitem ao Gênese interagir com o ambiente de forma controlada:

| Ferramenta | Descrição | Uso |
| :--- | :--- | :--- |
| `escrever_arquivo(caminho, conteudo)` | Escreve o `conteudo` em um arquivo no `workspace_genesis`. | Permite a criação de relatórios, scripts e outros artefatos. |
| `executar_git(comando)` | Executa um comando Git (ex: `add .`, `commit -m "..."`) no diretório raiz do projeto. | Permite que o Gênese versionar seu próprio trabalho. |
