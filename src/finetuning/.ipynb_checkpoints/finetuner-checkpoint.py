"""
src/finetuning/finetuner.py
fine-tuning module:
  - TrainingDataGenerator  (auto Q&A from chunks + KG, quality filter)
  - OpenAIFinetuner        (gpt-4o-mini cloud fine-tuning)
  - LoRAFinetuner          (local GPU, Mistral/LLaMA with QLoRA)
"""
from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from rich.console import Console
from rich.progress import track
from rich.table import Table

from src.config import get_settings

console = Console()

# ══════════════════════════════════════════════════════════════
#  Training Data Generator
# ══════════════════════════════════════════════════════════════

SYSTEM_PROMPT = ("You are a domain expert assistant. Answer questions accurately, "
                 "concisely, and factually.")

QA_GEN_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Generate {n} diverse Q&A pairs from the text.
Return ONLY valid JSON (no markdown):
{{"pairs":[{{"question":"...","answer":"..."}}]}}
Rules: answerable from chunk only, 2-4 sentence answers, varied question types, no yes/no questions."""),
    ("human", "Text:\n{chunk}"),
])

QUALITY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", 'Score 0.0–1.0. Return ONLY: {{"score":0.85,"reason":"..."}}'),
    ("human", "Question:{question}\nAnswer:{answer}\nContext:{context}"),
])

@dataclass
class TrainingSample:
    question: str
    answer: str
    context: str
    source: str
    quality_score: float = 1.0

class TrainingDataGenerator:
    def __init__(self, quality_threshold: float = 0.65):
        s = get_settings()
        self.quality_threshold = quality_threshold
        llm = ChatOpenAI(model=s.openai_chat_model, openai_api_key=s.openai_api_key,
                         temperature=0.7, response_format={"type": "json_object"})
        scorer = ChatOpenAI(model=s.openai_chat_model, openai_api_key=s.openai_api_key,
                            temperature=0, response_format={"type": "json_object"})
        self._gen   = QA_GEN_PROMPT | llm | StrOutputParser()
        self._score = QUALITY_PROMPT | scorer | StrOutputParser()

    def generate_from_chunks(self, chunks: List[Document], n_per_chunk: int = 3,
                              max_chunks: Optional[int] = None,
                              filter_quality: bool = True) -> List[TrainingSample]:
        target = chunks[:max_chunks] if max_chunks else chunks
        raw: List[TrainingSample] = []
        for chunk in track(target, description="[finetune] Generating Q&A..."):
            for pair in self._gen_pairs(chunk.page_content, n_per_chunk):
                raw.append(TrainingSample(question=pair["question"], answer=pair["answer"],
                                          context=chunk.page_content,
                                          source=chunk.metadata.get("source", "unknown")))
        return self._filter(raw) if filter_quality else raw

    def generate_from_kg(self, neo4j_client, n_pairs: int = 50) -> List[TrainingSample]:
        samples: List[TrainingSample] = []
        entities = neo4j_client.search_entities("", limit=30)
        for entity in entities[:20]:
            neighbors = neo4j_client.get_neighbors(entity["name"], depth=1, limit=5)
            if not neighbors: continue
            ctx = f"Entity: {entity['name']} ({entity['type']})\nRelationships:\n"
            for n in neighbors:
                ctx += f"  {' → '.join(n['relations'])} → {n['entity']}\n"
            for pair in self._gen_pairs(ctx, n=2):
                samples.append(TrainingSample(question=pair["question"], answer=pair["answer"],
                                              context=ctx, source="knowledge_graph"))
        return samples[:n_pairs]

    def generate_manual(self, jsonl_path: str) -> List[TrainingSample]:
        """Load a manually created JSONL file: {question, answer, context}"""
        samples = []
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line: continue
                d = json.loads(line)
                samples.append(TrainingSample(
                    question=d["question"], answer=d["answer"],
                    context=d.get("context", ""), source=d.get("source", "manual"),
                    quality_score=1.0))
        print(f"[finetune] Loaded {len(samples)} manual samples from {jsonl_path}")
        return samples

    def save_openai_jsonl(self, samples: List[TrainingSample], path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for s in samples:
                f.write(json.dumps({"messages": [
                    {"role": "system",    "content": SYSTEM_PROMPT},
                    {"role": "user",      "content": s.question},
                    {"role": "assistant", "content": s.answer},
                ]}) + "\n")
        print(f"[finetune] Saved {len(samples)} OpenAI samples → {path}")

    def save_alpaca_jsonl(self, samples: List[TrainingSample], path: str,
                          include_context: bool = True) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for s in samples:
                f.write(json.dumps({"instruction": s.question,
                                     "input":       s.context[:800] if include_context else "",
                                     "output":      s.answer}) + "\n")
        print(f"[finetune] Saved {len(samples)} Alpaca samples → {path}")

    def split(self, samples: List[TrainingSample],
              val_ratio: float = 0.1) -> tuple[List[TrainingSample], List[TrainingSample]]:
        random.shuffle(samples)
        split = max(1, int(len(samples) * (1 - val_ratio)))
        return samples[:split], samples[split:]

    def stats(self, samples: List[TrainingSample]) -> dict:
        if not samples: return {}
        return {
            "total": len(samples),
            "avg_q_len": round(sum(len(s.question) for s in samples) / len(samples)),
            "avg_a_len": round(sum(len(s.answer)   for s in samples) / len(samples)),
            "avg_quality": round(sum(s.quality_score for s in samples) / len(samples), 3),
            "sources": {s.source: sum(1 for x in samples if x.source == s.source) for s in samples},
        }

    def _gen_pairs(self, text: str, n: int) -> List[dict]:
        try:
            raw = self._gen.invoke({"chunk": text[:2500], "n": n})
            return [p for p in json.loads(raw).get("pairs", [])
                    if p.get("question") and p.get("answer")]
        except Exception as e:
            print(f"[data_gen] error: {e}")
            return []

    def _filter(self, samples: List[TrainingSample]) -> List[TrainingSample]:
        passed = []
        for s in track(samples, description="[finetune] Quality scoring..."):
            try:
                raw = self._score.invoke({"question": s.question,
                                          "answer":   s.answer,
                                          "context":  s.context[:1000]})
                score = float(json.loads(raw).get("score", 0))
                s.quality_score = score
                if score >= self.quality_threshold:
                    passed.append(s)
            except Exception:
                passed.append(s)
        print(f"[finetune] Quality: {len(passed)}/{len(samples)} passed")
        return passed

# ══════════════════════════════════════════════════════════════
#  OpenAI Fine-tuner
# ══════════════════════════════════════════════════════════════

class OpenAIFinetuner:
    BASE_MODEL = "gpt-4o-mini-2024-07-18"

    def __init__(self):
        from openai import OpenAI
        self.client = OpenAI(api_key=get_settings().openai_api_key)

    def validate(self, path: str) -> tuple[bool, str]:
        if not Path(path).exists(): return False, f"Not found: {path}"
        count = errors = 0
        with open(path) as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line: continue
                try:
                    r = json.loads(line)
                    roles = [m.get("role") for m in r.get("messages", [])]
                    if "user" not in roles or "assistant" not in roles:
                        errors += 1
                    count += 1
                except json.JSONDecodeError:
                    errors += 1
        if count < 10: return False, f"Too few samples: {count}"
        if errors:     return False, f"{errors} invalid lines"
        return True, f"Valid — {count} samples"

    def upload(self, path: str) -> str:
        with open(path, "rb") as f:
            r = self.client.files.create(file=f, purpose="fine-tune")
        console.print(f"[green]Uploaded: {r.id}[/green]")
        return r.id

    def create_job(self, train_id: str, val_id: Optional[str] = None,
                   n_epochs: int = 3, suffix: str = "rag"):
        kwargs = {"training_file": train_id, "model": self.BASE_MODEL,
                  "hyperparameters": {"n_epochs": n_epochs}, "suffix": suffix}
        if val_id: kwargs["validation_file"] = val_id
        job = self.client.fine_tuning.jobs.create(**kwargs)
        console.print(f"[green]Job created: {job.id}[/green]")
        return job

    def wait(self, job_id: str, poll: int = 30) -> Optional[str]:
        terminal = {"succeeded", "failed", "cancelled"}
        while True:
            job = self.client.fine_tuning.jobs.retrieve(job_id)
            console.print(f"[dim]Status: {job.status}[/dim]")
            if job.status in terminal: break
            time.sleep(poll)
        if job.status == "succeeded":
            Path("data").mkdir(exist_ok=True)
            Path("data/finetuned_model.txt").write_text(job.fine_tuned_model)
            console.print(f"[green]Model: {job.fine_tuned_model}[/green]")
            return job.fine_tuned_model
        return None

    def compare(self, model_id: str, questions: List[str]) -> None:
        t = Table(title="Base vs Fine-tuned")
        t.add_column("Question", max_width=30); t.add_column("Base", max_width=40); t.add_column("Fine-tuned", max_width=40)
        for q in questions[:5]:
            base = self._ask(self.BASE_MODEL, q)
            ft   = self._ask(model_id, q)
            t.add_row(q[:60], base[:120], ft[:120])
        console.print(t)

    def _ask(self, model: str, q: str) -> str:
        try:
            r = self.client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": q}],
                temperature=0, max_tokens=150)
            return r.choices[0].message.content or ""
        except Exception as e:
            return f"[error: {e}]"

# ══════════════════════════════════════════════════════════════
#  LoRA Fine-tuner
# ══════════════════════════════════════════════════════════════

@dataclass
class LoRAConfig:
    base_model: str = "mistralai/Mistral-7B-Instruct-v0.3"
    use_4bit: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list = field(default_factory=lambda: [
        "q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])
    output_dir: str = "./outputs/lora_model"
    num_epochs: int = 3
    per_device_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    max_seq_length: int = 2048
    save_steps: int = 100


class LoRAFinetuner:
    def __init__(self, config: LoRAConfig | None = None):
        self.config = config or LoRAConfig()

    def train(self, train_path: str, val_path: Optional[str] = None) -> None:
        self._check_gpu()
        model, tokenizer = self._load_model()
        model = self._apply_lora(model)
        train_ds = self._load_dataset(train_path)
        val_ds   = self._load_dataset(val_path) if val_path else None
        self._run(model, tokenizer, train_ds, val_ds)

    def merge_and_save(self, adapter_path: Optional[str] = None) -> str:
        from peft import AutoPeftModelForCausalLM
        import torch
        path = adapter_path or self.config.output_dir
        model = AutoPeftModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16, device_map="auto")
        merged_path = path + "_merged"
        model.merge_and_unload().save_pretrained(merged_path)
        console.print(f"[green]Merged → {merged_path}[/green]")
        return merged_path

    def push_to_hub(self, repo_id: str) -> None:
        from peft import AutoPeftModelForCausalLM
        from transformers import AutoTokenizer
        model = AutoPeftModelForCausalLM.from_pretrained(self.config.output_dir)
        tokenizer = AutoTokenizer.from_pretrained(self.config.output_dir)
        model.push_to_hub(repo_id); tokenizer.push_to_hub(repo_id)

    def _check_gpu(self):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("No GPU detected. LoRA requires CUDA. Use OpenAI fine-tuning for CPU.")
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        console.print(f"[cyan]GPU: {torch.cuda.get_device_name(0)} | {vram:.1f}GB VRAM[/cyan]")
        if vram < 6: raise RuntimeError(f"Insufficient VRAM ({vram:.1f}GB). Min 8GB for QLoRA.")

    def _load_model(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        tokenizer = AutoTokenizer.from_pretrained(self.config.base_model, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        if self.config.use_4bit:
            bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                     bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
            model = AutoModelForCausalLM.from_pretrained(self.config.base_model,
                                                          quantization_config=bnb, device_map="auto")
        else:
            model = AutoModelForCausalLM.from_pretrained(self.config.base_model,
                                                          torch_dtype=torch.float16, device_map="auto")
        return model, tokenizer

    def _apply_lora(self, model):
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
        if self.config.use_4bit: model = prepare_model_for_kbit_training(model)
        cfg = LoraConfig(r=self.config.lora_r, lora_alpha=self.config.lora_alpha,
                         target_modules=self.config.target_modules,
                         lora_dropout=self.config.lora_dropout, bias="none",
                         task_type=TaskType.CAUSAL_LM)
        model = get_peft_model(model, cfg)
        model.print_trainable_parameters()
        return model

    def _load_dataset(self, path: str):
        from datasets import Dataset
        records = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line: continue
                d = json.loads(line)
                ctx = d.get("input", "")
                text = (f"### Instruction:\n{d['instruction']}\n\n### Input:\n{ctx}\n\n### Response:\n{d['output']}"
                        if ctx else
                        f"### Instruction:\n{d['instruction']}\n\n### Response:\n{d['output']}")
                records.append({"text": text})
        return Dataset.from_list(records)

    def _run(self, model, tokenizer, train_ds, val_ds):
        from transformers import TrainingArguments
        from trl import SFTTrainer
        args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.per_device_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate, warmup_ratio=0.03,
            save_steps=self.config.save_steps, logging_steps=10, fp16=True,
            optim="paged_adamw_32bit" if self.config.use_4bit else "adamw_torch",
            evaluation_strategy="steps" if val_ds else "no",
            report_to="none",
        )
        trainer = SFTTrainer(model=model, tokenizer=tokenizer, train_dataset=train_ds,
                             eval_dataset=val_ds, dataset_text_field="text",
                             max_seq_length=self.config.max_seq_length, args=args)
        trainer.train()
        trainer.save_model(self.config.output_dir)
        tokenizer.save_pretrained(self.config.output_dir)
        console.print(f"[green]LoRA training complete → {self.config.output_dir}[/green]")
