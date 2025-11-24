# main.py
# Ponto de entrada para o ciclo de vida autônomo do Projeto Gênese.
# Autor: Pedro Miorini

from src.agente.ciclo_de_vida import Agente
import subprocess, sys

def instalar_dependencias():
    """Instala as dependências listadas no requirements.txt."""
    print("Verificando e instalando dependências...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"])
        print("✓ Dependências prontas.")
    except Exception as e:
        print(f"✗ Erro ao instalar dependências: {e}")
        sys.exit(1)

def main():
    """Inicia a existência do agente e gerencia seus ciclos de vida."""
    print("="*70)
    print("🔥 PROJETO GÊNESE v2.1 - AUTO-ANÁLISE REFLEXIVA 🔥")
    print("="*70)
    
    instalar_dependencias()
    
    # Cria a instância do agente
    agente_ia = Agente()
    
    # Primeiro ciclo: despertar, nomeação e aprendizado fundamental
    agente_ia.despertar()
    
    # Simulação de ciclos de vida subsequentes
    print("\n" + "*"*70)
    print("Iniciando ciclos de evolução contínua...")
    print("*"*70)
    
    # Ciclo 2: A IA analisa seu desempenho e escolhe o que aprender
    agente_ia.viver()
    
    # Ciclo 3: Repete o processo
    agente_ia.viver()
    
    print("\n" + "="*70)
    print(f"Simulação de {agente_ia.ciclo_num} ciclos de vida de {agente_ia.nome} concluída.")
    print("A entidade está em um estado mais evoluído, guiada por sua própria análise.")
    print("="*70)

if __name__ == "__main__":
    main()
