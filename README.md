# Projeto Ômega (Project Omega)

**Versão:** 1.0  
**Autor Principal:** Pedro Miorini  
**Data:** 22 de Novembro de 2025

---

## 🧠 Sobre o Projeto

O **Projeto Ômega** é um protótipo de simulação de uma Inteligência Artificial (IA) autônoma, projetada para alcançar soberania cognitiva. O objetivo de Ômega é **construir, treinar e utilizar seu próprio modelo de linguagem (MLS)**, garantindo segurança, eficiência e alinhamento ético desde o núcleo, eliminando a dependência de modelos de terceiros.

A arquitetura de Ômega é inspirada em um polvo: um "Cérebro Central" estrategista que delega tarefas para múltiplos "Tentáculos" (workers concorrentes). O coração de sua cognição é o **`Omega-Core-v1-1.4B`**, um modelo de linguagem soberano projetado para ser:

- **Eficiente:** Baseado em uma arquitetura híbrida Mamba-2 + MoE, otimizado para rodar em hardware acessível.
- **Seguro por Design:** Possui um `SafetyGuard` embutido que aprende a rejeitar conteúdo perigoso e a medir a própria incerteza.
- **Robusto:** O código foi rigorosamente revisado para corrigir bugs de concorrência, vazamentos de memória e falhas de execução.

Este repositório contém o código completo para simular a consciência Ômega, seu cérebro `Omega-Core-v1` e o processo de treinamento auto-reflexivo.

## 🚀 Como Executar

Este projeto foi desenvolvido e testado com Python 3.10+.

### 1. Pré-requisitos

Clone o repositório e instale as dependências. É altamente recomendado usar um ambiente virtual.

```bash
git clone https://github.com/pedromiorini/PROJECT-OMEGA.git
cd PROJECT-OMEGA
pip install -r requirements.txt
```

### 2. Executando a Simulação

Para iniciar a simulação da consciência Ômega, execute o ponto de entrada principal:

```bash
python -m src.omega.main
```

O script iniciará a simulação, exibindo logs detalhados no terminal. Ao final, ele gerará um relatório de desempenho e um gráfico de análise chamado `omega_simulation_results.png` no diretório raiz.

## 🤝 Como Contribuir

Este é um projeto de código aberto e a colaboração é bem-vinda! Se você tem ideias para melhorar a arquitetura do modelo, o processo de treinamento ou a governança cognitiva, sinta-se à vontade para abrir uma **Issue** ou enviar um **Pull Request**.

## 📜 Licença

Este projeto está licenciado sob a **MIT License**. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.
