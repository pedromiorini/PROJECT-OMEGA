# src/modelo/treinador.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset
from peft import get_peft_model, LoraConfig, TaskType
import os

class Treinador:
    """Encapsula a lógica de treinamento base e de habilidades (LoRA)."""
    def __init__(self, device):
        self.device = device

    def treinar_modelo_base(self, dataset: Dataset, output_path: str):
        """Realiza o fine-tuning completo de um modelo base."""
        print(f"🧠 Iniciando Treinamento Base em '{output_path}'...")
        model_name = "pierreguillou/gpt2-small-portuguese"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = model.config.eos_token_id
        
        model.to(self.device)

        def tokenize(e): return tokenizer(e["text"], truncation=True, max_length=128)
        tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])
        
        training_args = TrainingArguments(
            output_dir=f"{output_path}_results", num_train_epochs=3,
            per_device_train_batch_size=8, gradient_accumulation_steps=2,
            logging_steps=50, report_to="none", fp16=(self.device.type == 'cuda')
        )
        trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_dataset,
                          tokenizer=tokenizer, data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False))
        
        trainer.train()
        trainer.save_model(output_path)
        tokenizer.save_pretrained(output_path)
        print(f"✅ Treinamento Base concluído. Modelo salvo em '{output_path}'.")

    def treinar_habilidade_lora(self, modelo_base_path: str, dataset: Dataset, adapter_path: str):
        """Treina um adaptador LoRA sobre um modelo base."""
        print(f"✨ Iniciando Treinamento de Habilidade (LoRA) em '{adapter_path}'...")
        tokenizer = AutoTokenizer.from_pretrained(modelo_base_path)
        model = AutoModelForCausalLM.from_pretrained(modelo_base_path)
        model.to(self.device)

        for param in model.parameters(): param.requires_grad = False

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32, lora_dropout=0.05,
            bias="none", target_modules=["c_attn", "c_proj", "c_fc"]
        )
        model_peft = get_peft_model(model, lora_config)
        model_peft.print_trainable_parameters()

        def tokenize(e): return tokenizer(e["text"], truncation=True, max_length=128)
        tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])

        training_args = TrainingArguments(
            output_dir=f"{adapter_path}_results", num_train_epochs=10,
            per_device_train_batch_size=4, gradient_accumulation_steps=4,
            logging_steps=10, report_to="none", learning_rate=2e-4,
            fp16=(self.device.type == 'cuda')
        )
        trainer = Trainer(model=model_peft, args=training_args, train_dataset=tokenized_dataset,
                          tokenizer=tokenizer, data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False))

        trainer.train()
        model_peft.save_pretrained(adapter_path)
        print(f"✅ Treinamento de Habilidade concluído. Adaptador salvo em '{adapter_path}'.")
