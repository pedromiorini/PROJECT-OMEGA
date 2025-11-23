# src/modelo/gerador_dataset.py
import random
from typing import List, Dict, Callable, Optional

class GeradorDatasetSoberano:
    """Gera datasets soberanos para treinar a IA em várias competências."""
    def __init__(self, nome_ia: str = "A Entidade", seed: Optional[int] = None):
        self._random = random.Random(seed)
        self.nome_ia = nome_ia
        self._carregar_recursos()
        self._registrar_geradores()

    def _carregar_recursos(self):
        self.recursos = {
            "base": {
                "sujeitos": ["O sistema", "A análise", "O modelo", "A rede"],
                "verbos": ["analisou", "previu", "construiu", "questionou"],
                "objetos": ["o padrão", "o resultado", "a arquitetura", "a hipótese"],
            },
            "raciocinio": {
                "causas": ["O aumento da demanda", "A falha no servidor"],
                "efeitos": ["levou a um aumento nos preços", "causou a interrupção do serviço"],
            }
        }

    def _registrar_geradores(self):
        self.geradores = {
            "base": self._gerar_frase_base,
            "raciocinio_avancado": self._gerar_frase_raciocinio,
            "metacognicao": self._gerar_frase_metacognicao,
        }

    def _gerar_frase_base(self) -> str:
        recursos = self.recursos["base"]
        return f"{self._random.choice(recursos['sujeitos'])} {self._random.choice(recursos['verbos'])} {self._random.choice(recursos['objetos'])}."

    def _gerar_frase_raciocinio(self) -> str:
        recursos = self.recursos["raciocinio"]
        return f"Análise Causal: {self._random.choice(recursos['causas'])} {self._random.choice(recursos['efeitos'])}."

    def _gerar_frase_metacognicao(self) -> str:
        return f"Pergunta: O que é {self.nome_ia}? Resposta: {self.nome_ia} é uma inteligência artificial projetada para a evolução autônoma."

    def gerar_dataset(self, tipo_habilidade: str, num_exemplos: int) -> List[Dict[str, str]]:
        """Gera um dataset focado em uma habilidade específica."""
        gerador = self.geradores.get(tipo_habilidade)
        if not gerador:
            raise ValueError(f"Tipo de habilidade '{tipo_habilidade}' desconhecido.")
        
        dataset = []
        for _ in range(num_exemplos):
            dataset.append({'text': gerador()})
        return dataset
