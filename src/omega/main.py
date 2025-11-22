# src/omega/main.py
import sys
import signal
import logging
from .agent import Omega

def main():
    """Função principal com tratamento de erros e sinais para execução segura."""
    omega_instance = None
    try:
        omega_instance = Omega()
        
        def signal_handler(sig, frame):
            print("\n")
            if omega_instance:
                omega_instance.stop()
        
        signal.signal(signal.SIGINT, signal_handler)
        
        omega_instance.iniciar_simulacao()
        
        print("\n[FIM DA SIMULAÇÃO] Ômega completou seu ciclo de trabalho e análise.")
        
    except ValueError as ve:
        print(f"[ERRO FATAL NA INICIALIZAÇÃO] {ve}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n[INTERRUPÇÃO MANUAL] Simulação abortada pelo usuário.")
        if omega_instance:
            omega_instance.stop()
        sys.exit(0)
    except Exception as e:
        print(f"[ERRO INESPERADO] {e}")
        logging.exception("Erro crítico durante execução:")
        sys.exit(1)

if __name__ == "__main__":
    main()
