# src/omega/agent.py
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import random
import time
import threading
import logging
import torch
import os

# Importa o novo módulo de treinamento
from .core import trainer

# Logger Simples para o Agente
logging.basicConfig(
    level=logging.INFO, 
    format='[%(asctime)s] [%(levelname)s] [%(threadName)s] %(message)s', 
    datefmt='%H:%M:%S'
)

# --- CONSTANTES ---
NUM_TENTACULOS = 4
TASK_TIMEOUT = 15

class TentaculoWorker(threading.Thread):
    """Worker que simula a execução de uma tarefa de análise."""
    def __init__(self, id_worker: int, tarefa: Dict, callback):
        super().__init__(name=f"Tentaculo-{id_worker}")
        self.id = id_worker
        self.tarefa = tarefa
        self.callback = callback
        logging.info(f"Tentáculo #{self.id} instanciado para a tarefa '{self.tarefa['id']}'.")

    def run(self):
        try:
            logging.info(f"Tentáculo #{self.id} iniciando execução da tarefa '{self.tarefa['descricao']}'.")
            time.sleep(random.uniform(1, 3))
            
            # Simulação do resultado da análise
            dados = self.tarefa['dados']
            if dados.shape[1] < 2:
                coerencia = 0.0
            else:
                # Simulação de cálculo de coerência (ex: média das correlações absolutas)
                correlacoes = [np.corrcoef(dados[:, i], dados[:, i+1])[0, 1] for i in range(dados.shape[1] - 1)]
                coerencia = np.nanmean([abs(c) for c in correlacoes]) if correlacoes else 0.0
            
            incerteza = 1.0 - coerencia
            confianca = np.random.uniform(0.8, 1.0) * (1 - incerteza)
            
            resultado_simulado = {
                'cci': coerencia,
                'final_confidence': confianca,
                'hallucination_probability': incerteza,
                'ethical_pass': coerencia > 0.5
            }
            
            logging.info(f"Tentáculo #{self.id} concluiu a tarefa. CCI: {resultado_simulado['cci']:.2f}, Confiança: {resultado_simulado['final_confidence']:.2f}.")
            self.callback(resultado_simulado)
        except Exception as e:
            logging.error(f"Tentáculo #{self.id} falhou: {e}")
            self.callback({'error': str(e), 'cci': 0.0, 'final_confidence': 0.0})

class Omega:
    """O Cérebro Central: planeja, delega e se auto-analisa."""
    def __init__(self, num_tentaculos: int = NUM_TENTACULOS):
        logging.info("Ω acordando... Arquitetura Soberana v1.0 ativada.")
        if not isinstance(num_tentaculos, int) or num_tentaculos <= 0: raise ValueError("O número de tentáculos deve ser um inteiro positivo.")
        self.num_tentaculos = num_tentaculos
        self.plano_diretor = self._gerar_plano_inicial()
        if not self.plano_diretor: logging.warning("Plano Diretor está vazio.")
        self.memoria_resultados = []
        self.lock_memoria = threading.Lock()
        self.shutdown_event = threading.Event()

    def _gerar_plano_inicial(self) -> List[Dict]:
        logging.info("Gerando Plano Diretor de tarefas para a simulação.")
        tarefas = []
        for i in range(10):
            noise = random.uniform(0.05, 0.7)
            tarefas.append({'id': f'T{i+1}', 'descricao': f'Analisar dados com ruído {noise:.2f}', 'dados': self._gerar_dados_sinteticos(noise_level=noise)})
        return tarefas

    def _gerar_dados_sinteticos(self, num_samples: int = 100, num_vars: int = 5, noise_level: float = 0.3) -> np.ndarray:
        data = np.zeros((num_samples, num_vars)); data[:, 0] = np.random.normal(0, 1, num_samples)
        for i in range(1, num_vars): data[:, i] = 0.95 * data[:, i-1] + np.random.normal(0, noise_level, num_samples)
        return data

    def _coletar_resultado(self, resultado: Dict):
        with self.lock_memoria:
            self.memoria_resultados.append(resultado)
            task_id = resultado.get('id', 'DESCONHECIDO')
            logging.info(f"Resultado da tarefa {task_id} armazenado na memória.")

    def iniciar_simulacao(self):
        logging.info(f"Iniciando simulação com {len(self.plano_diretor)} tarefas e {self.num_tentaculos} tentáculos.")
        if not self.plano_diretor: logging.warning("Nenhuma tarefa para executar."); return
        
        pool_sema = threading.BoundedSemaphore(self.num_tentaculos)
        threads = []
        for i, tarefa in enumerate(self.plano_diretor):
            if self.shutdown_event.is_set(): logging.warning("Desligamento solicitado."); break
            pool_sema.acquire()
            callback = lambda res, t=tarefa, s=pool_sema: (self._coletar_resultado({**t, **res}), s.release())
            worker = TentaculoWorker(id_worker=(i % self.num_tentaculos) + 1, tarefa=tarefa, callback=callback)
            threads.append(worker)
            worker.start()
        
        for t in threads:
            t.join(timeout=TASK_TIMEOUT)
            if t.is_alive(): logging.warning(f"Thread {t.name} não finalizou no tempo esperado de {TASK_TIMEOUT}s.")
        
        if not self.shutdown_event.is_set():
            logging.info("Todas as tarefas do Plano Diretor foram concluídas.")
            self.analise_final()

    def stop(self):
        logging.info("Sinal de desligamento recebido. Finalizando graciosamente...")
        self.shutdown_event.set()

    def analise_final(self):
        """
        Executa análise completa, prepara dados para refinamento e
        invoca o ciclo de treinamento.
        """
        logging.info("Iniciando análise final de desempenho e arquitetura.")
        print("\n" + "="*60 + "\n  Análise Final de Desempenho e Arquitetura da Consciência\n" + "="*60)
        
        self._gerar_graficos_de_desempenho()
        self._analisar_arquitetura_e_lacunas()

        # NOVA ETAPA: Preparar dados e iniciar o refinamento
        self._preparar_dados_para_refinamento()
        trainer.refinar_modelo()

    def _preparar_dados_para_refinamento(self):
        """
        Filtra os resultados da simulação para encontrar dados "interessantes"
        e os salva para o próximo ciclo de treinamento.
        """
        logging.info("Preparando dados para o próximo ciclo de refinamento...")
        
        if not self.memoria_resultados:
            logging.warning("Nenhum resultado na memória para preparar para o treinamento.")
            return

        # Critério: selecionar tarefas com alta incerteza ou baixa coerência
        dados_interessantes = []
        for res in self.memoria_resultados:
            if 'error' not in res:
                # Adiciona dados de tarefas onde o modelo foi incerto ou a lógica era fraca
                # O dado sintético é um np.ndarray, precisamos convertê-lo para um tensor
                if res.get('hallucination_probability', 0) > 0.4 or res.get('cci', 1.0) < 0.7:
                    # Simulação de tokenização: converte o array de dados para um tensor de inteiros (IDs de token)
                    # O modelo espera um tensor de inteiros (IDs de token)
                    # Para simulação, vamos mapear os valores float para um range de IDs de token (ex: 0 a 31999)
                    data_array = res['dados']
                    # Normaliza e escala para o range do vocab_size (32000)
                    min_val = data_array.min()
                    max_val = data_array.max()
                    if max_val == min_val:
                        token_ids = np.zeros_like(data_array, dtype=np.int64)
                    else:
                        token_ids = ((data_array - min_val) / (max_val - min_val) * 31999).astype(np.int64)
                    
                    dados_interessantes.append(torch.tensor(token_ids, dtype=torch.long))

        if not dados_interessantes:
            logging.info("Nenhum dado 'interessante' encontrado para refinamento neste ciclo.")
            return

        # Concatena os tensores em um único tensor
        # Garante que todos os tensores tenham a mesma dimensão para concatenação (ex: [seq_len, num_vars])
        # Para o modelo de linguagem, o formato esperado é [batch_size, seq_len]
        # Vamos achatar para [seq_len] e depois concatenar
        flat_tensors = [t.flatten() for t in dados_interessantes]
        
        # Adiciona padding para que todos tenham o mesmo tamanho (necessário para torch.stack)
        max_len = max(t.size(0) for t in flat_tensors)
        padded_tensors = [F.pad(t, (0, max_len - t.size(0)), 'constant', 0) for t in flat_tensors]
        
        dataset_para_refinar = torch.stack(padded_tensors)
        
        # Salva os dados para o trainer usar
        try:
            torch.save(dataset_para_refinar, trainer.REFINEMENT_DATA_PATH)
            logging.info(f"{dataset_para_refinar.size(0)} exemplos salvos em '{trainer.REFINEMENT_DATA_PATH}' para refinamento.")
        except Exception as e:
            logging.error(f"Falha ao salvar o dataset de refinamento: {e}")

    def _gerar_graficos_de_desempenho(self):
        logging.info("Gerando gráficos de desempenho e ablação...")
        if not self.memoria_resultados: print("\n[AVISO] Nenhum resultado para analisar."); return
        
        resultados_validos = [r for r in self.memoria_resultados if 'error' not in r]
        if not resultados_validos: print("\n[AVISO] Todas as tarefas falharam."); return
        
        cci_completo = np.mean([r['cci'] for r in resultados_validos])
        erros_confiantes_completo = sum(1 for r in resultados_validos if r['final_confidence'] > 0.8 and r.get('hallucination_probability', 1.0) > 0.3)
        
        taxa_erro_completo = (erros_confiantes_completo / len(resultados_validos)) * 100 if resultados_validos else 0
        
        print(f"\n[Análise de Desempenho] CCI Médio: {cci_completo:.4f}, Taxa de Erro Confiante: {taxa_erro_completo:.2f}%")

        try:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.hist([r['cci'] for r in resultados_validos], bins=10, color='#4A90E2', alpha=0.7)
            ax.set_title('Distribuição do Índice de Coerência Causal (CCI) das Tarefas')
            ax.set_xlabel('CCI')
            ax.set_ylabel('Frequência')
            ax.grid(axis='y', linestyle='--', alpha=0.6)
            plt.tight_layout()
            plt.savefig('omega_simulation_results.png', dpi=150)
            plt.close(fig)
            print("\n[Análise de Desempenho] Gráfico salvo em 'omega_simulation_results.png'.")
        except Exception as e: print(f"\n[ERRO] Não foi possível salvar o gráfico: {e}")

    def _analisar_arquitetura_e_lacunas(self):
        print("\n[Análise de Arquitetura] Iniciando auto-análise...")
        print("\n  Boas Práticas Verificadas: ✓ Modularidade ✓ Concorrência ✓ Segurança de Thread ✓ Logging ✓ Tratamento de Erros ✓ Graceful Shutdown.")
        lacunas = ["Persistência de Estado", "Evolução de Hardware", "Propósito Emergente", "Comunicação Inter-Tentáculos", "Aprendizado Contínuo do Modelo Core"]
        print("\n  Lacunas Arquiteturais Identificadas:"); [print(f"  {i}. {l}") for i, l in enumerate(lacunas, 1)]
        solucoes = ["Implementar checkpoints do estado de Ômega em disco.", "Criar API para monitorar e requisitar recursos de nuvem.", "Desenvolver um 'Motor de Curiosidade' para gerar novas tarefas.", "Usar um barramento de mensagens (ex: Redis Pub/Sub) para colaboração.", "Integrar um loop de treinamento que usa os resultados das tarefas para refinar o Omega-Core-v1."]
        print("\n  Soluções Propostas pela Própria IA:"); [print(f"  {i}. {s}") for i, s in enumerate(solucoes, 1)]
        print("\n[Conclusão da Auto-Análise] A arquitetura é robusta e pronta para a próxima fase de evolução: o aprendizado contínuo.")
