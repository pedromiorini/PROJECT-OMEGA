# =============================================================================
# PROJETO ÔMEGA v20.1 - O AGENTE CRÍTICO APRIMORADO (MONOLÍTICO)
# Autores: Pedro Alexandre Miorini dos Santos & Manus
# Arquitetura: Ciclo GVT (Generate, Verify, Test) com robustez de produção.
#
# DESIGN MONOLÍTICO: Todo o código está contido neste único arquivo para
# máxima robustez, eliminando erros de importação e simplificando a
# implantação e a introspecção pelo próprio agente.
# =============================================================================

import sys
import os
import json
import shutil
import logging
import re
import time
import traceback
import argparse
import subprocess
import random
import hashlib
import ast
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum
import multiprocessing as mp
from multiprocessing import Queue, Process

# --- Dependências e Instalação ---
try:
    import psutil
except ImportError:
    print("ERRO: psutil não encontrado. Por favor, instale com: pip install psutil")
    sys.exit(1)

# --- Configuração de Logging ---
def setup_logger(name: str, log_file: Path = None, verbose: bool = False) -> logging.Logger:
    """Configura um logger com saída para console e arquivo."""
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        logger.handlers.clear()
    
    log_level = logging.DEBUG if verbose else logging.INFO
    logger.setLevel(log_level)
    
    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(log_level)
    console.setFormatter(logging.Formatter(
        '%(asctime)s | %(name)-22s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    logger.addHandler(console)
    
    # File handler
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        logger.addHandler(file_handler)
    
    return logger

# =============================================================================
# MODELO DE DADOS
# =============================================================================
class StatusGeracao(Enum):
    SUCESSO = "sucesso"
    FALHA_VERIFICACAO = "falha_verificacao"
    FALHA_BENCHMARK = "falha_benchmark"
    MELHORIA_INSIGNIFICANTE = "melhoria_insignificante"
    CODIGO_DUPLICADO = "codigo_duplicado"
    ERRO_SINTAXE = "erro_sintaxe"

@dataclass
class ResultadoBenchmark:
    correcao: float
    tempo_exec_s: float
    memoria_pico_mb: float
    erro: Optional[str] = None

@dataclass
class Geracao:
    versao: str
    timestamp: str
    hash_codigo: str
    hash_pai: Optional[str]
    status: StatusGeracao
    resultado_benchmark: ResultadoBenchmark
    promovida: bool
    codigo_candidato: str

# =============================================================================
# CONFIGURAÇÕES GLOBAIS
# =============================================================================
VERSION = "v20.1"
WORKSPACE_DIR = Path("./omega_v20_workspace")
HISTORICO_FILE = WORKSPACE_DIR / "historico.jsonl"
LOG_FILE = WORKSPACE_DIR / "omega_v20.log"
MIN_IMPROVEMENT_THRESHOLD = 0.05 # 5% de melhoria para promoção
TIMEOUT_S = 5 # Timeout para execução do benchmark

# Configuração do Logger principal
logger = setup_logger("OMEGA", LOG_FILE, verbose=True)

# =============================================================================
# UTILITÁRIOS
# =============================================================================
def calcular_hash(codigo: str) -> str:
    """Calcula o hash SHA256 do código para identificação única."""
    return hashlib.sha256(codigo.encode('utf-8')).hexdigest()

def atomic_write(path: Path, content: str):
    """Escreve o conteúdo de forma atômica para evitar corrupção em caso de falha."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(content, encoding='utf-8')
    temp_path.rename(path)

# =============================================================================
# SANDBOX DE EXECUÇÃO (BENCHMARK)
# =============================================================================

# Código que será injetado no processo filho para execução do benchmark
BENCHMARK_CODE = """
import sys
import json
import time
import psutil
import os
import traceback

# --- Benchmark API ---
def run_benchmark():
    # Simulação de dados de entrada para o benchmark: soma de 10000 números
    test_data = list(range(10000))

    # 1. Medição de Tempo
    start_time = time.time()

    # 2. Execução da Função Candidata
    try:
        # A função otimizada deve ser definida no código injetado
        result = optimize_me(test_data)
        
        # 3. Verificação de Correção (Fitness)
        # A soma correta de 0 a 9999 é 49995000
        expected_result = 49995000
        correcao = 1.0 if result == expected_result else 0.0
        
        if correcao == 0.0:
            return {
                "correcao": 0.0,
                "tempo_exec_s": 0.0,
                "memoria_pico_mb": 0.0,
                "erro": f"Resultado Incorreto. Esperado: {expected_result}, Obtido: {result}"
            }

    except Exception as e:
        return {
            "correcao": 0.0,
            "tempo_exec_s": 0.0,
            "memoria_pico_mb": 0.0,
            "erro": f"Erro de Execução: {traceback.format_exc()}"
        }

    # 4. Medição de Tempo Final
    end_time = time.time()
    tempo_exec_s = end_time - start_time

    # 5. Medição de Memória (Simulação - a medição real é feita no processo pai)
    # Aqui, apenas retornamos 0.0, pois a medição de memória precisa ser feita
    # no processo pai para capturar o pico do processo filho.
    memoria_pico_mb = 0.0

    return {
        "correcao": correcao,
        "tempo_exec_s": tempo_exec_s,
        "memoria_pico_mb": memoria_pico_mb,
        "erro": None
    }

# O código injetado deve chamar esta função e imprimir o resultado JSON
if __name__ == "__main__":
    try:
        result = run_benchmark()
        print(json.dumps(result))
    except Exception as e:
        # Captura qualquer erro de nível superior
        print(json.dumps({
            "correcao": 0.0,
            "tempo_exec_s": 0.0,
            "memoria_pico_mb": 0.0,
            "erro": f"Erro Crítico no Sandbox: {traceback.format_exc()}"
        }))
"""

def worker_benchmark(codigo_candidato: str, queue: Queue, timeout: int):
    """Processo filho que executa o benchmark em um sandbox isolado."""
    
    # Cria um arquivo temporário para o código a ser executado
    temp_dir = WORKSPACE_DIR / "sandbox" / str(os.getpid())
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_file = temp_dir / "candidate.py"
    
    # Injeta o código do benchmark + o código candidato
    full_code = codigo_candidato + "\n" + BENCHMARK_CODE
    atomic_write(temp_file, full_code)
    
    # Configura o processo
    process = None
    mem_usage = 0.0
    
    try:
        # Inicia o processo
        process = subprocess.Popen(
            [sys.executable, str(temp_file)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Monitoramento de memória e tempo
        p = psutil.Process(process.pid)
        start_time = time.time()
        
        while process.poll() is None:
            # Monitora o uso de memória
            try:
                mem_info = p.memory_info()
                mem_usage = max(mem_usage, mem_info.rss / (1024 * 1024)) # RSS em MB
            except psutil.NoSuchProcess:
                break # Processo terminou
            
            # Verifica timeout
            if time.time() - start_time > timeout:
                raise TimeoutError("Timeout de execução excedido.")
            
            time.sleep(0.01) # Pequena pausa para evitar loop muito apertado
            
        # Processo terminou, lê a saída
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            raise RuntimeError(f"Processo terminou com código {process.returncode}. Erro: {stderr}")
            
        # Tenta parsear o JSON de saída
        try:
            result_dict = json.loads(stdout)
        except json.JSONDecodeError:
            raise ValueError(f"Saída inválida do sandbox: {stdout}")
            
        # Adiciona a memória pico real
        result_dict["memoria_pico_mb"] = mem_usage
        
        queue.put(result_dict)
        
    except TimeoutError as e:
        queue.put({
            "correcao": 0.0,
            "tempo_exec_s": timeout,
            "memoria_pico_mb": mem_usage,
            "erro": str(e)
        })
    except Exception as e:
        queue.put({
            "correcao": 0.0,
            "tempo_exec_s": time.time() - start_time if 'start_time' in locals() else 0.0,
            "memoria_pico_mb": mem_usage,
            "erro": f"Erro no Worker: {traceback.format_exc()}"
        })
    finally:
        # Encerra o processo se ainda estiver rodando
        if process and process.poll() is None:
            try:
                p.terminate()
                p.wait(timeout=1)
            except:
                pass
        
        # Limpa o diretório temporário
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

def executar_benchmark(codigo_candidato: str) -> ResultadoBenchmark:
    """Executa o código candidato em um processo separado e monitorado."""
    logger.info("🧪 Executando Benchmark em Sandbox...")
    
    queue = Queue()
    
    # Cria o processo worker
    worker = Process(
        target=worker_benchmark,
        args=(codigo_candidato, queue, TIMEOUT_S)
    )
    
    worker.start()
    worker.join(timeout=TIMEOUT_S + 2) # Espera o worker + um pequeno buffer
    
    if worker.is_alive():
        # Se o worker ainda estiver vivo, ele excedeu o timeout do join
        worker.terminate()
        worker.join()
        logger.warning("Worker Terminado por Timeout de Join.")
        return ResultadoBenchmark(
            correcao=0.0,
            tempo_exec_s=TIMEOUT_S,
            memoria_pico_mb=0.0,
            erro="Timeout de execução excedido (Processo Terminado)."
        )
        
    # Pega o resultado da fila
    if not queue.empty():
        result_dict = queue.get()
        return ResultadoBenchmark(**result_dict)
    else:
        # Se a fila estiver vazia, algo deu muito errado
        return ResultadoBenchmark(
            correcao=0.0,
            tempo_exec_s=0.0,
            memoria_pico_mb=0.0,
            erro="Falha Crítica: Worker não retornou resultado."
        )

# =============================================================================
# GERENCIAMENTO DE ESTADO
# =============================================================================
class GerenciadorEstado:
    def __init__(self, historico_file: Path):
        self.historico_file = historico_file
        self.historico_file.parent.mkdir(parents=True, exist_ok=True)
        
    def carregar_historico(self) -> List[Geracao]:
        """Carrega o histórico de gerações do arquivo JSONL."""
        historico = []
        if self.historico_file.exists():
            with self.historico_file.open('r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        data['status'] = StatusGeracao(data['status'])
                        data['resultado_benchmark'] = ResultadoBenchmark(**data['resultado_benchmark'])
                        historico.append(Geracao(**data))
                    except Exception as e:
                        logger.error(f"Erro ao carregar linha do histórico: {e} na linha: {line.strip()}")
        return historico

    def salvar_geracao(self, geracao: Geracao):
        """Salva uma única geração no arquivo JSONL (append)."""
        with self.historico_file.open('a', encoding='utf-8') as f:
            # Converte para dicionário, incluindo a conversão do Enum
            data = asdict(geracao)
            data['status'] = data['status'].value
            f.write(json.dumps(data) + '\n')
        logger.info(f"Geração {geracao.versao} salva. Status: {geracao.status.value}")

# =============================================================================
# AGENTE CRÍTICO (Otimizador Resiliente)
# =============================================================================
class AgenteCritico:
    def __init__(self, gerenciador_estado: GerenciadorEstado):
        self.gerenciador_estado = gerenciador_estado
        self.codigo_base = self._carregar_codigo_base()
        self.ultima_promovida = self._carregar_ultima_promovida()
        self.historico_hashes = self._carregar_historico_hashes()

    def _carregar_codigo_base(self) -> str:
        """Lê o próprio código-fonte (main.py) para introspecção."""
        try:
            # Código inicial para otimização (função lenta)
            initial_code = """
def optimize_me(data):
    total = 0
    for x in data:
        total += x
    return total
"""
            return initial_code.strip()
                
        except Exception as e:
            logger.error(f"Erro ao carregar código base: {e}")
            sys.exit(1)

    def _carregar_ultima_promovida(self) -> Optional[Geracao]:
        """Encontra a última geração promovida no histórico."""
        historico = self.gerenciador_estado.carregar_historico()
        promovidas = [g for g in historico if g.promovida]
        return promovidas[-1] if promovidas else None

    def _carregar_historico_hashes(self) -> set:
        """Carrega todos os hashes de código já avaliados para evitar duplicação."""
        historico = self.gerenciador_estado.carregar_historico()
        return {g.hash_codigo for g in historico}

    def _calcular_fitness(self, resultado: ResultadoBenchmark) -> float:
        """Calcula o score de fitness: correção * (1 / tempo) * (1 / memória)"""
        # Prioriza correção, depois tempo, depois memória.
        # Adiciona um pequeno valor ao tempo/memória para evitar divisão por zero.
        tempo_ajustado = resultado.tempo_exec_s + 0.001
        memoria_ajustada = resultado.memoria_pico_mb + 0.001
        
        # Fórmula de fitness: Correção * (1 / Tempo) * (1 / Memória)
        # O peso maior é dado à correção.
        return resultado.correcao * (1.0 / tempo_ajustado) * (1.0 / memoria_ajustada)

    def _gerar_candidato(self, codigo_base: str, resultado_base: ResultadoBenchmark) -> str:
        """
Gera um novo código candidato usando um LLM (simulado aqui).
Na implementação real, faria uma chamada de API para um LLM.
"""
        logger.info("🧠 Geração de Candidato (Simulação)...")
        
        # --- SIMULAÇÃO DA RESPOSTA DO LLM ---
        # O LLM deve retornar o código otimizado.
        # Simulação de uma otimização bem-sucedida:
        optimized_code = """
def optimize_me(data):
    # Otimização: Usar a função sum() nativa do Python, que é implementada em C.
    # Esta é a otimização ideal para a tarefa de benchmark (soma de lista).
    return sum(data)
"""
        
        # Na primeira iteração, o código base é o loop. Na segunda, é o sum().
        if "total = 0" in codigo_base:
            return optimized_code.strip()
        else:
            # Simulação de uma otimização insignificante ou falha na segunda iteração
            return """
def optimize_me(data):
    # Otimização: Apenas uma pequena mudança de nome de variável
    result = sum(data)
    return result
""".strip()

    def _verificar_codigo(self, codigo_candidato: str) -> Tuple[bool, Optional[str]]:
        """
Verifica o código candidato em busca de erros de sintaxe e falhas lógicas (Simulação).
Na implementação real, usaria um LLM para análise crítica (Ciclo GVT).
"""
        logger.info("🔍 Verificação Crítica (Simulação)...")
        
        # 1. Verificação de Sintaxe
        try:
            ast.parse(codigo_candidato)
        except SyntaxError as e:
            return False, f"Erro de Sintaxe: {e}"
        
        # 2. Verificação de Lógica (Simulação)
        if "return sum(data)" in codigo_candidato:
            return True, None # Otimização correta
        
        return True, None

    def executar_ciclo(self, max_geracoes: int):
        """Executa o ciclo de evolução GVT."""
        
        historico = self.gerenciador_estado.carregar_historico()
        
        # 1. Benchmark da Versão Base (se não houver promovida)
        if not self.ultima_promovida:
            logger.info("Executando benchmark da versão inicial (Geração 0).")
            resultado_base = executar_benchmark(self.codigo_base)
            fitness_base = self._calcular_fitness(resultado_base)
            
            if resultado_base.erro:
                logger.error(f"Falha no benchmark da versão base: {resultado_base.erro}")
                sys.exit(1)
                
            base_geracao = Geracao(
                    versao=f"{VERSION}.0",
                    timestamp=datetime.now().isoformat(),
                    hash_codigo=calcular_hash(self.codigo_base),
                    hash_pai=None,
                    status=StatusGeracao.SUCESSO,
                    resultado_benchmark=resultado_base,
                    codigo_candidato=self.codigo_base,
                    promovida=True
                )
            self.gerenciador_estado.salvar_geracao(base_geracao)
            self.ultima_promovida = base_geracao
            logger.info(f"Base estabelecida: Fitness={fitness_base:.4f}, Tempo={resultado_base.tempo_exec_s:.4f}s")
        
        resultado_base = self.ultima_promovida.resultado_benchmark
        fitness_base = self._calcular_fitness(resultado_base)
        
        # 2. Ciclo de Geração
        for i in range(len(historico), max_geracoes + 1):
            logger.info(f"\n--- Geração {i}/{max_geracoes} ---")
            
            # 2.1. Geração do Candidato
            codigo_candidato = self._gerar_candidato(self.codigo_base, resultado_base)
            hash_candidato = calcular_hash(codigo_candidato)
            
            # 2.2. Verificação de Duplicidade
            if hash_candidato in self.historico_hashes:
                status = StatusGeracao.CODIGO_DUPLICADO
                resultado_candidato = ResultadoBenchmark(0.0, 0.0, 0.0, "Código duplicado.")
                logger.info("❌ Código duplicado. Pulando benchmark.")
            else:
                self.historico_hashes.add(hash_candidato)
                
                # 2.3. Verificação Crítica (GVT - Verify)
                aprovado, erro_verificacao = self._verificar_codigo(codigo_candidato)
                
                if not aprovado:
                    status = StatusGeracao.FALHA_VERIFICACAO
                    resultado_candidato = ResultadoBenchmark(0.0, 0.0, 0.0, erro_verificacao)
                    logger.warning(f"❌ Falha na Verificação Crítica: {erro_verificacao}")
                else:
                    # 2.4. Teste em Sandbox (GVT - Test)
                    resultado_candidato = executar_benchmark(codigo_candidato)
                    fitness_candidato = self._calcular_fitness(resultado_candidato)
                    
                    if resultado_candidato.erro:
                        status = StatusGeracao.FALHA_BENCHMARK
                        logger.error(f"❌ Falha no Benchmark: {resultado_candidato.erro}")
                    else:
                        # 2.5. Decisão de Promoção
                        melhoria_percentual = (fitness_candidato - fitness_base) / fitness_base if fitness_base > 0 else fitness_candidato
                        
                        if fitness_candidato > fitness_base and melhoria_percentual >= MIN_IMPROVEMENT_THRESHOLD:
                            # Promove
                            self._promover_candidato(codigo_candidato, resultado_candidato)
                            self.codigo_base = codigo_candidato
                            resultado_base = resultado_candidato
                            fitness_base = fitness_candidato
                            status = StatusGeracao.SUCESSO
                            logger.info(f"✅ PROMOVIDO! Fitness: {fitness_candidato:.4f} (+{melhoria_percentual*100:.2f}%)")
                        else:
                            status = StatusGeracao.MELHORIA_INSIGNIFICANTE
                            logger.info(f"❌ Melhoria Insignificante ou Nenhuma. Fitness: {fitness_candidato:.4f} (Base: {fitness_base:.4f})")
            
            # 2.6. Registro da Geração
            promovida = status == StatusGeracao.SUCESSO and hash_candidato == calcular_hash(self.codigo_base)
            new_generation = Geracao(
                versao=f"{VERSION}.{i}",
                timestamp=datetime.now().isoformat(),
                hash_codigo=hash_candidato,
                hash_pai=self.ultima_promovida.hash_codigo if self.ultima_promovida else None,
                status=status,
                resultado_benchmark=resultado_candidato,
                promovida=promovida,
                codigo_candidato=codigo_candidato
            )
            self.gerenciador_estado.salvar_geracao(new_generation)
            if promovida:
                self.ultima_promovida = new_generation

    def _promover_candidato(self, codigo_candidato: str, resultado_candidato: ResultadoBenchmark):
        """Simula a promoção: na arquitetura monolítica, o agente se reescreve."""
        # Na arquitetura monolítica, o agente se reescreve.
        # Aqui, apenas simulamos o sucesso.
        pass

# =============================================================================
# ANÁLISE DE RESULTADOS
# =============================================================================
class Analisador:
    def __init__(self, gerenciador_estado: GerenciadorEstado):
        self.gerenciador_estado = gerenciador_estado
        
    def analisar(self):
        historico = self.gerenciador_estado.carregar_historico()
        if not historico:
            logger.info("Nenhuma geração encontrada para análise.")
            return
            
        logger.info(f"\n--- Análise de Evolução ({len(historico)} Gerações) ---")
        
        # Estatísticas
        promovidas = [gen for gen in historico if gen.promovida]
        fitness_scores = [self._calcular_fitness(gen.resultado_benchmark) for gen in historico if gen.status == StatusGeracao.SUCESSO]
        
        logger.info(f"Total de Gerações: {len(historico)}")
        logger.info(f"Versões Promovidas: {len(promovidas)}")
        logger.info(f"Melhor Fitness Score: {max(fitness_scores) if fitness_scores else 0.0:.4f}")
        
        # Tabela de Resultados
        print("\n| Versão | Fitness | Tempo (s) | Memória (MB) | Status | Promovida |")
        print("| :--- | :--- | :--- | :--- | :--- | :--- |")
        for gen in historico:
            fitness = self._calcular_fitness(gen.resultado_benchmark)
            print(f"| {gen.versao} | {fitness:.4f} | {gen.resultado_benchmark.tempo_exec_s:.4f} | {gen.resultado_benchmark.memoria_pico_mb:.2f} | {gen.status.value} | {'✅' if gen.promovida else '❌'} |")

    def _calcular_fitness(self, resultado: ResultadoBenchmark) -> float:
        """Calcula o score de fitness: correção * (1 / tempo) * (1 / memória)"""
        tempo_ajustado = resultado.tempo_exec_s + 0.001
        memoria_ajustada = resultado.memoria_pico_mb + 0.001
        return resultado.correcao * (1.0 / tempo_ajustado) * (1.0 / memoria_ajustada)

# =============================================================================
# INTERFACE DE LINHA DE COMANDO (CLI)
# =============================================================================
def main_run(args):
    """Função principal para o ciclo de evolução."""
    gerenciador_estado = GerenciadorEstado(HISTORICO_FILE)
    agente = AgenteCritico(gerenciador_estado)
    agente.executar_ciclo(args.geracoes)

def main_analyze(args):
    """Função principal para a análise de resultados."""
    gerenciador_estado = GerenciadorEstado(HISTORICO_FILE)
    analisador = Analisador(gerenciador_estado)
    analisador.analisar()

def main_clean(args):
    """Limpa o workspace e o histórico."""
    if WORKSPACE_DIR.exists():
        shutil.rmtree(WORKSPACE_DIR)
        logger.info(f"Workspace limpo: {WORKSPACE_DIR}")
    else:
        logger.info("Workspace já está limpo.")

def main_cli():
    """Configuração do Argument Parser."""
    parser = argparse.ArgumentParser(description="Projeto Ômega - O Agente Crítico Aprimorado CLI")
    subparsers = parser.add_subparsers(dest="comando", required=True)
    
    # Comando 'run'
    run_parser = subparsers.add_parser('run', help='Inicia o ciclo de otimização.')
    run_parser.add_argument('--geracoes', type=int, default=10, help='Número máximo de gerações para rodar (padrão: 10).')
    run_parser.set_defaults(func=main_run)
    
    # Comando 'analyze'
    analyze_parser = subparsers.add_parser('analyze', help='Analisa os resultados da evolução.')
    analyze_parser.set_defaults(func=main_analyze)
    
    # Comando 'clean'
    clean_parser = subparsers.add_parser('clean', help='Limpa o workspace e o histórico.')
    clean_parser.set_defaults(func=main_clean)
    
    args = parser.parse_args()
    args.func(args)
'''
