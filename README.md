# Projeto Ômega v20.1 - O Agente Crítico Aprimorado

**Um sistema de Inteligência Artificial que se auto-otimiza, projetado para resiliência, aprendizado contínuo e evolução mensurável, agora consolidado em uma arquitetura monolítica robusta.**

[![Status: Produção-Ready](https://img.shields.io/badge/status-production--ready-green.svg)](https://github.com/pedromiorini/PROJECT-OMEGA)
[![Licença: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

---

## 🧬 Visão Geral

O Projeto Ômega nasceu de uma pergunta fundamental: **"Pode uma IA não apenas resolver problemas, mas melhorar a si mesma de forma autônoma e segura?"**

A v20.1, "O Agente Crítico Aprimorado", é a nossa resposta mais madura. Este não é apenas um Large Language Model (LLM) envolto em um script. É um **sistema operacional para a evolução da IA**, um agente que implementa um ciclo perpétuo de auto-análise, otimização e validação, tudo contido em um único e robusto arquivo `main.py`.

O agente inicia com uma versão de si mesmo, mede sua performance (fitness) através de um rigoroso benchmark, gera uma nova versão "candidata" com melhorias, submete-a a uma verificação crítica interna e, se aprovada e provadamente superior, ela é promovida para se tornar a nova base para a próxima geração de evolução.

## ✨ Inovações Arquiteturais da v20.1

Esta versão representa a culminação de dezenas de iterações, aprendendo com cada falha para construir um sistema robusto e pronto para produção.

-   **Ciclo GVT (Generate, Verify, Test):** Inspirado no DeepSeek-Math-V2, o agente agora possui um "crítico interno". Ele primeiro gera uma solução, depois a verifica em busca de falhas lógicas e, só então, a testa em benchmark, resultando em um código de maior qualidade.
-   **Sandbox Multiplataforma:** Usa `multiprocessing` + `psutil` para isolar a execução de código candidato, impondo limites estritos de CPU, memória e timeout.
-   **Persistência e Recuperação:** O estado da evolução (histórico, fitness) é salvo em JSON, permitindo que o agente retome seu trabalho após uma interrupção.
-   **Sistema de Cache e Rollback:** Evita reavaliar soluções duplicadas e reverte para a última versão estável em caso de falha catastrófica.
-   **Design Monolítico:** Toda a lógica está contida em `main.py`, eliminando erros de importação e simplificando a implantação e a introspecção pelo próprio agente.
-   **CLI Profissional:** Uma interface de linha de comando completa (`run`, `analyze`, `clean`) com argumentos documentados para controle total do operador.

## 🚀 Como Funciona: O Ciclo de Vida

1.  **Introspecção:** O agente lê seu próprio código-fonte (`main.py`).
2.  **Benchmark Base:** Mede o "fitness" (correção, velocidade, memória) da sua versão atual.
3.  **Geração:** Envia seu código para um cérebro de IA (ex: Claude, GPT-4) para gerar uma versão otimizada.
4.  **Verificação:** O código candidato é analisado por um "revisor de código" de IA em busca de falhas lógicas.
5.  **Teste em Sandbox:** Se aprovado na verificação, o candidato é executado em um sandbox seguro e passa pelo mesmo benchmark rigoroso.
6.  **Decisão de Promoção:** Se o fitness do candidato for significativamente maior, ele é "promovido" e se torna a nova versão ativa.
7.  **Persistência:** O resultado da geração é salvo no histórico.
8.  **Repetição:** O ciclo recomeça.

## 🛠️ Uso

### Pré-requisitos
- Python 3.9+
- `pip install -r requirements.txt`

### Executando o Ciclo de Otimização
Para iniciar o ciclo de vida do agente com 10 gerações:
```bash
python main.py run --geracoes 10
```

### Analisando os Resultados
Para ver uma análise estatística da evolução a partir do arquivo `historico.json`:
```bash
python main.py analyze
```

## 📜 Nossa Jornada e Filosofia

Este projeto é o resultado de uma longa jornada. Começamos com a "Evolução Cega" (EGGROLL), falhamos, aprendemos e pivotamos para o "Despertar" com a arquitetura GTR (Generate, Test, Refine). A v20.1 é a materialização dessa filosofia: a inteligência não emerge da aleatoriedade, mas de um ciclo disciplinado de **raciocínio, verificação, experimentação e correção**.

## 🤝 Contribuições

Este é um projeto vivo. Contribuições são bem-vindas. Sinta-se à vontade para abrir uma *issue* para discutir novas estratégias de otimização, melhorias no sandbox ou novas tarefas de benchmark.

## 📄 Licença

Este projeto é licenciado sob a Licença MIT. Veja o arquivo [LICENSE] para mais detalhes.