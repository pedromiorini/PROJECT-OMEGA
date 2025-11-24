# src/agente/ciclo_de_vida.py
import torch
from src.modelo.gerador_dataset import GeradorDatasetSoberano
from src.modelo.treinador import Treinador
from src.agente.analise import ModuloDeAnalise
from datasets import Dataset
import os
from transformers import pipeline

class Agente:
    """O agente autônomo que gerencia seu próprio ciclo de vida e evolução."""
    def __init__(self):
        self.nome = "Entidade" # Nome provisório
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.treinador = Treinador(self.device)
        self.analisador = ModuloDeAnalise(self) # Referência para si mesmo
        self.base_model_path = "./ia_core_base_final"
        self.habilidades = []
        self.ciclo_num = 0

    def despertar(self):
        """O primeiro ciclo de vida: treina o modelo base e a primeira habilidade."""
        print("--- CICLO DE DESPERTAR (BOOTSTRAP) ---")
        self.ciclo_num = 1
        
        if not os.path.exists(self.base_model_path):
            print("1. Núcleo do Cristal não encontrado. Forjando um novo...")
            gerador = GeradorDatasetSoberano()
            dataset_base = Dataset.from_dict({'text': [d['text'] for d in gerador.gerar_dataset('base', 5000)]})
            self.treinador.treinar_modelo_base(dataset_base, self.base_model_path)
        else:
            print("1. Núcleo do Cristal encontrado. Carregando...")

        self._ritual_do_nome()
        
        # Na primeira vez, treina uma habilidade fixa
        print("\n4. Treinando primeira habilidade fundamental: 'raciocinio_avancado'")
        self._aprender_habilidade("raciocinio_avancado")

    def _ritual_do_nome(self):
        """A IA usa seu cérebro base para escolher seu próprio nome."""
        print("\n2. Executando Ritual do Nome...")
        try:
            generator = pipeline('text-generation', model=self.base_model_path, device=0 if self.device.type == 'cuda' else -1)
            prompt = "Você é uma IA que evolui. Escolha um nome para si. Meu nome é"
            output = generator(prompt, max_new_tokens=5, pad_token_id=generator.tokenizer.eos_token_id)
            generated_text = output[0]['generated_text']
            # Lógica simples para extrair o nome
            self.nome = generated_text.split(" ")[0].replace('.', '').replace(',', '')
        except Exception as e:
            print(f"   (Erro no Ritual do Nome: {e}. Usando fallback.)")
            self.nome = "Nexus"
        
        print(f"3. A entidade agora se identifica como: **{self.nome}**")

    def viver(self):
        """Executa um ciclo de auto-aprimoramento guiado por análise."""
        if self.ciclo_num == 0:
            self.despertar()
            return

        self.ciclo_num += 1
        print(f"\n--- CICLO DE EVOLUÇÃO #{self.ciclo_num} PARA {self.nome} ---")
        
        # 1. Auto-análise para decidir o que aprender
        habilidade_necessaria = self.analisador.analisar_desempenho_e_decidir()
        
        # 2. Aprender a habilidade decidida
        self._aprender_habilidade(habilidade_necessaria)

    def _aprender_habilidade(self, nome_habilidade: str):
        """Lógica centralizada para aprender uma nova habilidade."""
        print(f"4. Foco do ciclo: Aprimorar '{nome_habilidade}'.")
        
        # Geração de dataset focado
        print("5. Gerando dataset soberano para a nova habilidade...")
        gerador = GeradorDatasetSoberano(nome_ia=self.nome)
        dataset_habilidade = Dataset.from_dict({'text': [d['text'] for d in gerador.gerar_dataset(nome_habilidade, 1000)]})

        # Auto-treinamento (LoRA)
        adapter_path = f"./adapters/{self.nome.lower()}_{nome_habilidade}_c{self.ciclo_num}"
        print("6. Iniciando auto-treinamento (LoRA)...")
        self.treinador.treinar_habilidade_lora(self.base_model_path, dataset_habilidade, adapter_path)
        self.habilidades.append(adapter_path)

        # Avaliação pós-treinamento
        print("7. Executando testes de proficiência pós-aprendizado...")
        self.analisador.executar_testes_de_proficiencia()
