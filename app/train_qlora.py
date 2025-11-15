"""
Fine-tune small language models using QLoRA for IELTS scoring
Supports: Qwen 1.5B, Phi-2, Gemma 2B
Optimized for 4GB GPU with small batch sizes
"""
import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import load_dataset, Dataset
import json
from typing import Optional


class IELTSModelTrainer:
    """QLoRA trainer for IELTS scoring models"""
    
    # Supported models optimized for 4GB GPU
    SUPPORTED_MODELS = {
        "qwen": "Qwen/Qwen-1.5B",
        "phi2": "microsoft/phi-2",
        "gemma": "google/gemma-2b",
    }
    
    def __init__(
        self,
        model_name: str = "qwen",
        output_dir: str = "./models/ielts-finetuned",
        dataset_path: Optional[str] = None,
    ):
        self.model_name = model_name
        self.output_dir = output_dir
        self.dataset_path = dataset_path or "./data/ielts_training_data.jsonl"
        
        # Get model path
        if model_name in self.SUPPORTED_MODELS:
            self.base_model = self.SUPPORTED_MODELS[model_name]
        else:
            self.base_model = model_name
            
    def load_model_and_tokenizer(self):
        """Load model with 4-bit quantization for memory efficiency"""
        
        # 4-bit quantization config for QLoRA
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model,
            trust_remote_code=True,
        )
        
        # Set pad token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Load model with quantization
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        
        # Prepare model for k-bit training
        self.model = prepare_model_for_kbit_training(self.model)
        
        print(f"✓ Loaded {self.base_model} with 4-bit quantization")
        
    def configure_lora(self):
        """Configure LoRA adapter for efficient fine-tuning"""
        
        lora_config = LoraConfig(
            r=8,  # LoRA rank
            lora_alpha=16,  # LoRA scaling
            target_modules=[
                "q_proj",
                "k_proj", 
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
        return lora_config
        
    def prepare_dataset(self):
        """Load and prepare IELTS training dataset"""
        
        # Load dataset from JSONL
        if os.path.exists(self.dataset_path):
            with open(self.dataset_path, "r", encoding="utf-8") as f:
                data = [json.loads(line) for line in f]
            dataset = Dataset.from_list(data)
        else:
            # Create sample dataset if none exists
            print("⚠ No dataset found, creating sample data...")
            data = self._create_sample_dataset()
            dataset = Dataset.from_list(data)
            
        # Format dataset for training
        def format_prompt(example):
            prompt = f"""### Instruction:
Score the following IELTS speaking response. Provide scores for Fluency, Lexical Resource, Grammatical Range, and Pronunciation.

### Transcript:
{example['transcript']}

### Response:
{json.dumps(example['scores'], indent=2)}
"""
            return {"text": prompt}
            
        dataset = dataset.map(format_prompt)
        
        # Split dataset
        dataset = dataset.train_test_split(test_size=0.1)
        
        print(f"✓ Prepared {len(dataset['train'])} training samples")
        return dataset
        
    def _create_sample_dataset(self):
        """Create sample IELTS training data"""
        return [
            {
                "transcript": "Well, I think that technology has greatly influenced our daily lives. For example, smartphones allow us to stay connected with friends and family.",
                "scores": {
                    "fluency": 7.0,
                    "lexical_resource": 6.5,
                    "grammatical_range": 7.0,
                    "pronunciation": 7.0,
                }
            },
            {
                "transcript": "Uh, I like to, um, spend time with my family on weekends. We usually go to park or sometimes watch movie.",
                "scores": {
                    "fluency": 5.5,
                    "lexical_resource": 5.0,
                    "grammatical_range": 5.0,
                    "pronunciation": 6.0,
                }
            },
        ]
        
    def train(
        self,
        num_epochs: int = 3,
        batch_size: int = 1,  # Small for 4GB GPU
        gradient_accumulation_steps: int = 4,
        learning_rate: float = 2e-4,
        max_seq_length: int = 512,
    ):
        """Train the model with QLoRA"""
        
        # Load model and prepare LoRA
        self.load_model_and_tokenizer()
        lora_config = self.configure_lora()
        
        # Prepare dataset
        dataset = self.prepare_dataset()
        
        # Training arguments optimized for 4GB GPU
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            logging_steps=10,
            save_steps=100,
            eval_steps=100,
            evaluation_strategy="steps",
            save_total_limit=2,
            load_best_model_at_end=True,
            warmup_steps=50,
            fp16=True,  # Mixed precision for memory efficiency
            optim="paged_adamw_8bit",  # 8-bit optimizer
            gradient_checkpointing=True,  # Save memory
            max_grad_norm=0.3,
            report_to="none",  # Disable wandb/tensorboard
        )
        
        # Initialize SFTTrainer
        trainer = SFTTrainer(
            model=self.model,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            peft_config=lora_config,
            dataset_text_field="text",
            max_seq_length=max_seq_length,
            tokenizer=self.tokenizer,
            args=training_args,
        )
        
        # Start training
        print("🚀 Starting training...")
        trainer.train()
        
        # Save final model
        trainer.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        
        print(f"✓ Training complete! Model saved to {self.output_dir}")
        
        return trainer
        

def main():
    """Main training function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fine-tune IELTS scoring model with QLoRA")
    parser.add_argument(
        "--model",
        type=str,
        default="qwen",
        choices=["qwen", "phi2", "gemma"],
        help="Base model to fine-tune",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="./data/ielts_training_data.jsonl",
        help="Path to training dataset",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./models/ielts-finetuned",
        help="Output directory for trained model",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Training batch size",
    )
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = IELTSModelTrainer(
        model_name=args.model,
        output_dir=args.output,
        dataset_path=args.dataset,
    )
    
    # Train model
    trainer.train(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
