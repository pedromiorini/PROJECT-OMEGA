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
    """Inicia a existência do agente."""
    print("="*70)
    print("🔥 PROJETO GÊNESE v2.0 - O CICLO DE AUTOPOIESE 🔥")
    print("="*70)
    
    instalar_dependencias()
    
    # Cria a instância do agente
    agente_ia = Agente()
    
    # Primeiro ciclo: despertar e nomeação
    agente_ia.despertar()
    
    # Ciclos de vida subsequentes para aprendizado contínuo
    # (aqui simulamos apenas um ciclo, mas poderia ser um loop infinito)
    agente_ia.viver()
    
    print("\n" + "="*70)
    print(f"Ciclo de vida de {agente_ia.nome} concluído. A entidade está mais evoluída.")
    print("="*70)

if __name__ == "__main__":
    main()
