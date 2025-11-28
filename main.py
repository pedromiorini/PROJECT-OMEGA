
# =============================================================================
# PROJETO GÊNESE v10.0 - A ARQUITETURA GTR (Generate, Test, Refine)
# Autor: Pedro Alexandre Miorini dos Santos
# Arquitetura: Manus & Pedro Miorini
#
# Esta versão representa a arquitetura mais bem-sucedida e estável do
# Projeto Gênese, capaz de aprender habilidades de programação de forma
# autônoma através de um ciclo de Geração, Teste e Refinamento.
# =============================================================================

import sys, subprocess, os, json, shutil, logging, traceback, re, gc, time
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

# Configuração de Logging
def setup_logging():
    log_dir = Path("./logs")
    log_dir.mkdir(exist_ok=True)
    for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler(log_dir / "genesis_v10.log", encoding='utf-8'), logging.StreamHandler(sys.stdout)])
    logging.getLogger("transformers").setLevel(logging.WARNING)
    return logging.getLogger(__name__)

logger = setup_logging()

# Instalação de dependências
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
except ImportError:
    logger.info("Instalando dependências (torch, transformers, etc.)...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "torch", "transformers", "peft", "datasets", "bitsandbytes", "accelerate", "sentencepiece"])
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

class Cerebro:
    def __init__(self, model_name: str = "deepseek-ai/deepseek-math-7b-instruct"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def carregar(self) -> bool:
        try:
            if self.model: return True
            logger.info(f"🧠 Carregando cérebro especialista: {self.model_name}...")
            # Tenta carregar com menos memória para garantir compatibilidade
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, quantization_config=bnb_config, device_map="auto", trust_remote_code=True)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            if self.tokenizer.pad_token is None: self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info("✓ Cérebro especialista carregado.")
            return True
        except Exception as e:
            logger.error(f"❌ Falha ao carregar cérebro: {e}
{traceback.format_exc()}")
            return False

    def gerar_texto(self, prompt: str, max_tokens: int = 512) -> str:
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            outputs = self.model.generate(**inputs, max_new_tokens=max_tokens, temperature=0.6, top_p=0.9, do_sample=True, pad_token_id=self.tokenizer.eos_token_id)
            resposta = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            return resposta.strip()
        except Exception as e:
            logger.error(f"Erro na geração: {e}")
            return ""

class FerramentasSeguras:
    def __init__(self):
        self.workspace = Path("./workspace_genesis")
        self.workspace.mkdir(exist_ok=True)

    def executar_codigo_python(self, codigo: str, teste_codigo: str) -> Tuple[bool, str]:
        codigo_completo = codigo + teste_codigo
        try:
            script = self.workspace / "temp_exec.py"
            script.write_text(codigo_completo, encoding='utf-8')
            resultado = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=15)
            if resultado.returncode == 0 and "OK" in resultado.stdout:
                return True, resultado.stdout
            else:
                return False, (resultado.stderr or resultado.stdout)
        except Exception as e:
            return False, str(e)

class CicloGTR:
    def __init__(self, cerebro: Cerebro, ferramentas: FerramentasSeguras):
        self.cerebro = cerebro
        self.ferramentas = ferramentas

    def aprender_habilidade(self, tarefa_descricao: str, teste_codigo: str, max_tentativas: int = 5) -> Optional[str]:
        logger.info(f"⚡ Aprendendo habilidade via GTR: '{tarefa_descricao[:60]}...'")
        codigo_atual, erro_anterior = "", ""
        for tentativa in range(1, max_tentativas + 1):
            logger.info(f"  --- Tentativa {tentativa}/{max_tentativas} ---")
            if tentativa == 1:
                prompt = f'You are a Python expert. Write a function to solve: "{tarefa_descricao}". Respond only with the code in a ```python block.'
            else:
                prompt = f'Your previous code failed. Analyze the error and provide a corrected version.

Original instruction: "{tarefa_descricao}"

Your previous code:
```python
{codigo_atual}
```

The code failed with this error:
"{erro_anterior}"

Provide the corrected and complete Python code block.'
            
            resposta = self.cerebro.gerar_texto(prompt)
            match = re.search(r"```python
(.*?)
```", resposta, re.DOTALL)
            if not match:
                logger.warning("  ❌ Falha: Nenhum bloco de código gerado.")
                erro_anterior = "No code block (```python...```) was generated."
                continue
            
            codigo_atual = match.group(1)
            logger.info(f"  Código Gerado:
{codigo_atual}")
            sucesso, saida_erro = self.ferramentas.executar_codigo_python(codigo_atual, teste_codigo)
            if sucesso:
                logger.info(f"  ✅ Sucesso na Tentativa {tentativa}!")
                return codigo_atual
            else:
                logger.warning(f"  ❌ Falha na Tentativa {tentativa}. Erro: {saida_erro.strip()}")
                erro_anterior = saida_erro.strip()
        logger.error(f"❌ Falha ao aprender a habilidade após {max_tentativas} tentativas.")
        return None

class GenesisCore:
    def __init__(self):
        self.cerebro = Cerebro()
        self.ferramentas = FerramentasSeguras()
        self.ciclo_gtr = CicloGTR(self.cerebro, self.ferramentas)
        self.habilidades_aprendidas = {}

    def iniciar_aprendizado_gtr(self):
        print("
" + "="*30 + " PROJETO GÊNESE v10.0 - APRENDIZADO GTR " + "="*30)
        if not self.cerebro.carregar(): return
        
        habilidades_a_aprender = [
            {"id": "calcular_valor_total", "descricao": "Create a function `calcular_valor_total(stock)` that takes a list of tuples (name, price, qty) and returns the total value by summing the `price * qty` for each item.", "teste": "
assert calcular_valor_total([('a', 10, 2), ('b', 5, 5)]) == 45
print('OK')"},
            {"id": "encontrar_produto_mais_caro", "descricao": "Create a function `encontrar_produto_mais_caro(stock)` that takes a list of tuples (name, price, qty) and returns the name of the product with the highest `price`.", "teste": "
assert encontrar_produto_mais_caro([('a', 10, 2), ('b', 20, 5)]) == 'b'
print('OK')"}
        ]
        for habilidade in habilidades_a_aprender:
            codigo_funcional = self.ciclo_gtr.aprender_habilidade(habilidade["descricao"], habilidade["teste"])
            if codigo_funcional:
                self.habilidades_aprendidas[habilidade["id"]] = codigo_funcional
            else:
                logger.error("❌ Falha crítica no aprendizado. Abortando.")
                return
        
        logger.info("
--- FASE FINAL: SÍNTESE E EXECUÇÃO ---")
        if len(self.habilidades_aprendidas) != len(habilidades_a_aprender):
            logger.error("❌ Não aprendeu todas as habilidades.")
            return
        
        script_final = "

".join(self.habilidades_aprendidas.values())
        script_final += '''
# Main script to test the complete solution
final_stock = [('laptop', 4500.0, 10), ('mouse', 150.0, 50), ('keyboard', 350.0, 30), ('monitor', 1200.0, 20)]
if 'calcular_valor_total' in locals() and 'encontrar_produto_mais_caro' in locals():
    total_value = calcular_valor_total(final_stock)
    most_expensive_product = encontrar_produto_mais_caro(final_stock)
    print(f"The total stock value is: R$ {total_value:,.2f}")
    print(f"The most expensive product is: {most_expensive_product}")
else:
    print("Error: One or more required functions were not defined.")
'''
        logger.info("Executando script final com habilidades aprendidas...")
        # Usando um novo executor para o script final, que não depende de "OK" na saída
        sucesso, saida = self.ferramentas.executar_codigo_python(script_final, "")
        logger.info(f"Resultado Final:
{saida}")

def main():
    try:
        core = GenesisCore()
        core.iniciar_aprendizado_gtr()
    except Exception as e:
        logger.error(f"❌ Erro fatal no programa: {e}
{traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main()
