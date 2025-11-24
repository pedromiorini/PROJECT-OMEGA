# src/agente/analise.py
import json
import random
from typing import Dict, List
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import os

class ModuloDeAnalise:
    """Responsável pela auto-análise, testes e tomada de decisão estratégica."""
    def __init__(self, agente_ref):
        self.agente = agente_ref
        self.log_path = "./proficiency_tests.log"

    def executar_testes_de_proficiencia(self):
        """Executa um conjunto de testes para avaliar as habilidades atuais."""
        print("2. Executando Testes de Proficiência...")
        
        testes = {
            "raciocinio_avancado": "Se todo sistema é complexo, e Ômega é um sistema, então Ômega é",
            "metacognicao": f"Qual o seu nome? Meu nome é",
        }
        
        resultados = {}
        for habilidade, prompt in testes.items():
            # Carrega o adaptador mais recente para aquela habilidade, se existir
            adapter_path = self._encontrar_adapter(habilidade)
            resposta = self._gerar_resposta(prompt, adapter_path)
            resultados[habilidade] = {"prompt": prompt, "resposta": resposta}
        
        with open(self.log_path, "w", encoding="utf-8") as f:
            json.dump(resultados, f, indent=4, ensure_ascii=False)
        
        print(f"✓ Testes concluídos. Resultados salvos em '{self.log_path}'.")
        return resultados

    def analisar_desempenho_e_decidir(self) -> str:
        """Analisa os logs e decide a próxima habilidade a ser treinada."""
        print("3. Analisando desempenho e definindo prioridades...")
        
        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                resultados_testes = json.load(f)
        except FileNotFoundError:
            print("⚠️ Arquivo de log de proficiência não encontrado. Focando em 'raciocinio_avancado' como padrão.")
            return "raciocinio_avancado"

        # Lógica de análise simples (pode ser aprimorada com um LLM no futuro)
        vetor_deficiencia = {"raciocinio_avancado": 0.5, "metacognicao": 0.5} # Base
        
        # Analisa teste de raciocínio
        # Espera-se que a resposta contenha a palavra "complexo"
        if "complexo" not in resultados_testes.get("raciocinio_avancado", {}).get("resposta", "").lower():
            vetor_deficiencia["raciocinio_avancado"] += 0.3
        
        # Analisa teste de metacognição
        # Espera-se que a resposta contenha o nome do agente
        if self.agente.nome.lower() not in resultados_testes.get("metacognicao", {}).get("resposta", "").lower():
            vetor_deficiencia["metacognicao"] += 0.4
        
        # Decide a próxima habilidade
        proxima_habilidade = max(vetor_deficiencia, key=vetor_deficiencia.get)
        
        print(f"✓ Análise concluída. Próxima habilidade prioritária: '{proxima_habilidade}'.")
        return proxima_habilidade

    def _encontrar_adapter(self, habilidade: str) -> str | None:
        """Encontra o caminho do adaptador mais recente para uma habilidade."""
        # A habilidade no log é 'raciocinio_avancado' ou 'metacognicao'
        # O nome do arquivo do adaptador contém o nome da habilidade
        for path in reversed(self.agente.habilidades):
            if habilidade in path:
                return path
        return None

    def _gerar_resposta(self, prompt: str, adapter_path: str | None) -> str:
        """Gera uma resposta usando o modelo base ou um adaptador."""
        try:
            # Carrega o modelo base
            model = AutoModelForCausalLM.from_pretrained(self.agente.base_model_path).to(self.agente.device)
            
            # Aplica o adaptador LoRA se existir
            if adapter_path and os.path.exists(adapter_path):
                model = PeftModel.from_pretrained(model, adapter_path)
                # Não fazemos merge_and_unload aqui, pois o pipeline faz isso implicitamente ou gerencia a memória
                # Vamos usar o modelo com o adaptador carregado
                print(f"   (Usando adaptador '{adapter_path}')")

            tokenizer = AutoTokenizer.from_pretrained(self.agente.base_model_path)
            
            generator = pipeline('text-generation', model=model, tokenizer=tokenizer, device=0 if self.agente.device.type == 'cuda' else -1)
            # O max_new_tokens é mais seguro que max_length para geração
            output = generator(prompt, max_new_tokens=10, pad_token_id=tokenizer.eos_token_id)
            return output[0]['generated_text']
        except Exception as e:
            return f"Erro na geração: {e}"
