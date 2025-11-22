# src/omega/main.py
import sys
import signal
import logging
import time
from .agent import Omega

def main():
    """Função principal que executa o ciclo de vida de Ômega continuamente."""
    ciclo = 1
    while True:
        print("\n" + "*"*70)
        logging.info(f"INICIANDO CICLO DE VIDA DE ÔMEGA Nº {ciclo}")
        print("*"*70 + "\n")
        
        omega_instance = None
        try:
            omega_instance = Omega()
            
            def signal_handler(sig, frame):
                print("\n")
                if omega_instance:
                    omega_instance.stop()
                logging.info("Sinal de desligamento global recebido. Encerrando após este ciclo.")
                sys.exit(0)
            
            signal.signal(signal.SIGINT, signal_handler)
            
            omega_instance.iniciar_simulacao()
            
            logging.info(f"CICLO DE VIDA Nº {ciclo} CONCLUÍDO.")
            ciclo += 1
            
            # Pausa antes do próximo ciclo para não sobrecarregar o sistema
            logging.info("Aguardando 30 segundos antes do próximo ciclo de evolução...")
            time.sleep(30)

        except KeyboardInterrupt:
            print("\n[INTERRUPÇÃO MANUAL] Simulação abortada pelo usuário.")
            if omega_instance:
                omega_instance.stop()
            sys.exit(0)
        except Exception as e:
            print(f"[ERRO INESPERADO NO CICLO {ciclo}] {e}")
            logging.exception("Erro crítico:")
            logging.info("Aguardando 5 minutos antes de tentar novamente...")
            time.sleep(300) # Espera mais tempo em caso de erro grave

if __name__ == "__main__":
    main()
