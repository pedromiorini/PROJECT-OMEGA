
# PROJECT OMEGA (Gênese)

## Visão Geral

Este repositório representa o legado do **Projeto Gênese**, uma jornada para criar um agente de IA autônomo capaz de aprender, raciocinar e executar tarefas de programação. Este código, versão **v10.0**, implementa a arquitetura **GTR (Generate, Test, Refine)**, a mais bem-sucedida e estável que desenvolvemos.

## A Jornada

O projeto evoluiu através de várias versões, cada uma ensinando uma lição valiosa:
- **v1.0 - v8.0:** Luta inicial com bugs, setup e a exploração da evolução de pesos (EGGROLL). Concluímos que, embora fascinante, era a ferramenta errada para ensinar conceitos de programação do zero. O sinal de fitness era muito fraco e o espaço de busca, infinito.
- **v9.0:** A introdução do "Cérebro Especialista" (`deepseek-math-7b-instruct`), que nos deu a matéria-prima correta para o raciocínio.
- **v10.0 (Este código):** A grande pivotada. Abandonamos a evolução de pesos e implementamos o ciclo GTR. O Gênese aprendeu a gerar código, testá-lo, analisar o erro e se autocorrigir, simulando o fluxo de trabalho de um programador real. **Esta foi a nossa maior vitória.**
- **v11.0 - v12.0:** Tentativas de dar ao Gênese autonomia sobre seu ambiente (escrever arquivos, usar Git), que nos levaram a esta missão final de consolidação.

## Arquitetura v10.0: GTR (Generate, Test, Refine)

O `main.py` neste repositório implementa um ciclo de aprendizado de habilidades em três fases:

1.  **Geração:** O Cérebro Especialista recebe uma tarefa (ex: "Crie a função X") e gera uma primeira versão do código.
2.  **Teste:** O sistema executa o código gerado contra uma unidade de teste.
3.  **Refinamento:** Se o teste falhar, o sistema captura o `Traceback` do erro e o alimenta de volta ao Cérebro, instruindo-o a analisar o erro e gerar uma versão corrigida. O ciclo se repete até o sucesso.

Esta arquitetura provou ser robusta, resiliente e a base para qualquer desenvolvimento futuro do Projeto Gênese.

## Como Executar

1.  Certifique-se de ter um ambiente Python com as dependências listadas (principalmente `torch` e `transformers`).
2.  Execute o script: `python main.py`.
3.  O script irá carregar o modelo (pode exigir uma GPU com memória suficiente) e iniciar o ciclo de aprendizado GTR.

---
*Este repositório foi atualizado autonomamente como parte da missão v12.1, orquestrada por Manus e Pedro Miorini.*
