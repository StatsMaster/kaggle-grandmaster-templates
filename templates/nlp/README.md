# NLP: LoRA fine-tuning + RAG retriever

**The lesson (GenAI Era):** LLM Science Exam 2023 was won by Team H2O LLM Studio with *retrieval-augmented generation over Wikipedia + LoRA fine-tuning of 7B–13B LLMs + a 6-model ensemble*. The modern NLP playbook: don't train from scratch, don't even full-fine-tune — adapt a foundation model cheaply (LoRA) and ground it with retrieval.

**When to reach for it:** any text competition — classification, QA, generation with provided context.

**The pipeline:**
1. **LoRA fine-tune** (`lora_finetune.py`) — freeze the base model, train low-rank adapters. Fits on a single GPU; the H2O recipe framed each answer as a binary classifier.
2. **RAG retriever** (`rag_retriever.py`) — embed a corpus, build an index, retrieve top-k passages, stuff them into the prompt. Works with `sentence-transformers` if installed, TF-IDF if not.

**Files:**
- `lora_finetune.py` — peft LoRA + HF Trainer skeleton.
- `rag_retriever.py` — embed → index → retrieve → prompt template.

```bash
# TODO: MODEL_NAME + dataset in lora_finetune.py; CORPUS in rag_retriever.py
python templates/nlp/lora_finetune.py
python templates/nlp/rag_retriever.py
```
