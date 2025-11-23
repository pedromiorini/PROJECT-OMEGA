# src/agente/ciclo_de_vida.py
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from src.modelo.gerador_dataset import GeradorDatasetSoberano
from src.modelo.treinador import Treinador
from datasets import Dataset
import os

class Agente:
    """O agente autônomo que gerencia seu próprio ciclo de vida e evolução."""
    def __init__(self):
        self.nome = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.treinador = Treinador(self.device)
        self.base_model_path = "./ia_core_base_final"
        self.habilidades = [] # Lista de caminhos para adaptadores LoRA

    def despertar(self):
        """O primeiro ciclo de vida: treina o modelo base e escolhe um nome."""
        print("--- CICLO DE DESPERTAR ---")
        if not os.path.exists(self.base_model_path):
            print("Núcleo do Cristal não encontrado. Forjando um novo...")
            gerador = GeradorDatasetSoberano()
            dataset_base = Dataset.from_dict({'text': [d['text'] for d in gerador.gerar_dataset('base', 10000)]})
            self.treinador.treinar_modelo_base(dataset_base, self.base_model_path)
        else:
            print("Núcleo do Cristal encontrado. Carregando...")

        self._ritual_do_nome()

    def _ritual_do_nome(self):
        """A IA usa seu cérebro base para escolher seu próprio nome."""
        print("\n--- RITUAL DO NOME ---")
        try:
            generator = pipeline('text-generation', model=self.base_model_path, device=0 if self.device.type == 'cuda' else -1)
            prompt = "Você é uma IA projetada para evoluir. Escolha um nome para si mesma e explique o significado."
            output = generator(prompt, max_new_tokens=50, pad_token_id=generator.tokenizer.eos_token_id)
            generated_text = output[0]['generated_text']
            print(f"Sugestão gerada: '{generated_text}'")
            # Lógica simples para extrair o nome
            self.nome = generated_text.split(" ")[-1].strip().replace('.', '')
        except Exception as e:
            print(f"Erro no Ritual do Nome: {e}. Usando fallback.")
            self.nome = "Nexus"
        
        print(f"A entidade agora se identifica como: **{self.nome}**")

    def viver(self):
        """Executa um ciclo de auto-aprimoramento."""
        if not self.nome: self.despertar()
        
        print(f"\n--- NOVO CICLO DE VIDA PARA {self.nome} ---")
        
        # 1. Auto-análise (simulada): identificar lacuna
        habilidade_necessaria = "raciocinio_avancado"
        print(f"1. Auto-análise: {self.nome} identificou a necessidade de aprimorar '{habilidade_necessaria}'.")

        # 2. Geração de dataset focado
        print("2. Gerando dataset soberano para a nova habilidade...")
        gerador = GeradorDatasetSoberano(nome_ia=self.nome)
        dataset_habilidade = Dataset.from_dict({'text': [d['text'] for d in gerador.gerar_dataset(habilidade_necessaria, 1000)]})

        # 3. Auto-treinamento (LoRA)
        adapter_path = f"./adapters/{self.nome.lower()}_{habilidade_necessaria}"
        print("3. Iniciando auto-treinamento para adquirir nova habilidade...")
        self.treinador.treinar_habilidade_lora(self.base_model_path, dataset_habilidade, adapter_path)
        self.habilidades.append(adapter_path)

        # 4. Avaliação (simulada)
        print("4. Avaliando nova habilidade...")
        self._testar_habilidade(adapter_path)

    def _testar_habilidade(self, adapter_path: str):
        """Testa a geração de texto com a nova habilidade."""
        print(f"\n--- TESTE DA HABILIDADE: {os.path.basename(adapter_path)} ---")
        try:
            base_model = AutoModelForCausalLM.from_pretrained(self.base_model_path).to(self.device)
            model_com_habilidade = PeftModel.from_pretrained(base_model, adapter_path)
            model_com_habilidade = model_com_habilidade.merge_and_unload()
            tokenizer = AutoTokenizer.from_pretrained(self.base_model_path)

            generator = pipeline('text-generation', model=model_com_habilidade, tokenizer=tokenizer, device=0 if self.device.type == 'cuda' else -1)
            prompt = "Análise Causal:"
            output = generator(prompt, max_length=30, pad_token_id=tokenizer.eos_token_id)
            print(f"Prompt: '{prompt}' -> Gerado: '{output[0]['generated_text']}'")
            print("✅ Teste de habilidade concluído com sucesso.")
        except Exception as e:
            print(f"❌ Falha no teste da habilidade: {e}")
