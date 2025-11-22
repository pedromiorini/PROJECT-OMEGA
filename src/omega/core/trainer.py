# src/omega/core/trainer.py
import torch
import torch.optim as optim
import logging
import os
from .model import OmegaCoreModel, OmegaCoreConfig

# --- CONFIGURAÇÕES DE TREINAMENTO ---
LEARNING_RATE = 5e-5
NUM_EPOCHS = 3  # Número de vezes para treinar sobre o novo dataset
BATCH_SIZE = 4
MODEL_PATH = "omega_core_model.pt"
REFINEMENT_DATA_PATH = "refinement_dataset.pt"

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s')

def refinar_modelo():
    """
    Função principal para refinar o modelo Omega-Core-v1 com base
    nos dados coletados durante a simulação.
    """
    logging.info(">>> INICIANDO CICLO DE REFINAMENTO DO MODELO SOBERANO <<<")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Usando dispositivo: {device}")

    # 1. Verificar se há dados de refinamento
    if not os.path.exists(REFINEMENT_DATA_PATH):
        logging.warning("Nenhum dataset de refinamento encontrado. Pulando ciclo de treinamento.")
        return

    try:
        refinement_data = torch.load(REFINEMENT_DATA_PATH)
        if len(refinement_data) == 0:
            logging.warning("Dataset de refinamento está vazio. Pulando ciclo de treinamento.")
            return
        logging.info(f"Carregados {len(refinement_data)} exemplos do dataset de refinamento.")
    except Exception as e:
        logging.error(f"Falha ao carregar o dataset de refinamento: {e}")
        return

    # 2. Carregar o modelo
    config = OmegaCoreConfig()
    model = OmegaCoreModel(config).to(device)

    # Carrega o estado do modelo, se existir
    if os.path.exists(MODEL_PATH):
        logging.info(f"Carregando checkpoint do modelo de '{MODEL_PATH}'.")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    else:
        logging.warning("Nenhum checkpoint encontrado. Treinando a partir de pesos iniciais.")

    # 3. Configurar otimizador e loop de treinamento
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    model.train()

    logging.info(f"Iniciando treinamento por {NUM_EPOCHS} épocas...")
    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        num_batches = 0
        # Criar batches a partir dos dados
        for i in range(0, len(refinement_data), BATCH_SIZE):
            batch = refinement_data[i:i+BATCH_SIZE].to(device)
            if batch.shape[0] == 0: continue

            optimizer.zero_grad()
            
            # O modelo espera 'labels', então passamos o próprio batch
            logits, loss, safety_metrics = model(batch, labels=batch)
            
            if loss is not None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        logging.info(f"Época {epoch + 1}/{NUM_EPOCHS} concluída. Perda Média: {avg_loss:.4f}")

    # 4. Salvar o modelo refinado
    try:
        torch.save(model.state_dict(), MODEL_PATH)
        logging.info(f"Modelo refinado salvo com sucesso em '{MODEL_PATH}'.")
    except Exception as e:
        logging.error(f"Falha ao salvar o modelo refinado: {e}")

    # Opcional: Limpar o dataset de refinamento após o uso
    # os.remove(REFINEMENT_DATA_PATH)
    # logging.info("Dataset de refinamento utilizado e limpo.")

if __name__ == '__main__':
    # Este bloco permite executar o treinamento de forma independente
    # Ex: python -m src.omega.core.trainer
    refinar_modelo()
