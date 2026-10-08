# Task 2: Generative AI, QLoRA Fine-Tuning for AML Alert Triage

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_GITHUB_USERNAME/CDAZZDEV-MLE-YourName/blob/main/task2_genai/notebooks/Task2_AML_Triage_QLoRA.ipynb)

**Notebook:** [`notebooks/Task2_AML_Triage_QLoRA.ipynb`](notebooks/Task2_AML_Triage_QLoRA.ipynb) (run on a Colab T4) · **Model:** `https://huggingface.co/YOUR_HF_USERNAME/qwen2.5-1.5b-aml-alert-triage` · **Data:** [`data/`](data/)

## Use case
Triage of bank **transaction-monitoring alerts** (anti-money-laundering).
- **Input:** a rendered alert with the rule that fired, the customer's KYC profile, prior alerts, and a table of 3–10 transactions.
- **Output:** JSON with `typology` (11 classes), `risk_level`, grounded `red_flags` and `mitigating_factors`, `recommended_action` (`close_no_action` / `request_information` / `enhanced_due_diligence` / `escalate_sar`), and a `rationale`.
- **Correct:** valid JSON with the allowed labels, typology and action match the gold decision, and every cited fact exists in the alert.
- **Hallucinated:** any amount, total, document or counterparty that isn't in the alert.

The full problem statement is in the notebook header.

## Pipeline
| Stage | Module | Key design decisions |
|---|---|---|
| Seeds | `generate.build_seeds` (150 seeds → **131 validated cases**, split 105/13/13) | Stratified: 11 typologies, balanced with 22% benign false positives; 15 segments; 20 jurisdictions; 3 difficulty levels including *misleading-surface* cases. Reproducible (seed 2026). |
| Teacher | `generate.TeacherPool` | `openai/gpt-oss-120b` (Groq) is primary. When its daily quota runs out it rotates to `qwen/qwen3.8-27b` (Groq), then `nemotron-3-super-120b` (OpenRouter free). Every case records its teacher. **No teacher is the student.** Generation is resumable. |
| Quality gate | `generate.consistency_errors` | Pydantic schema, then: label must match the seed; risk and action must be coherent; and an **arithmetic grounding check**, where every figure must equal a transaction amount, the declared turnover, a subset-sum of transactions, or a regulatory threshold. `1.07M`-style abbreviations are expanded first. This caught a 10× error in the teacher's first trial batch. |
| Dataset | `dataset.py` | Conversational `prompt`/`completion` JSONL (system, user, assistant), rendered with Qwen's own chat template at train time. Stratified 80/10/10 split. Diversity report: length distribution, label/segment/keyword frequencies, distinct-n, all-pairs 5-gram Jaccard near-duplicate scan. |
| Training | `train.py` | QLoRA: 4-bit NF4 + double quantisation, fp16 compute, LoRA r=16/α=32 on all 7 linear projections, completion-only loss, cosine LR 2e-4, 3 epochs, per-epoch validation, best checkpoint kept, `merge_and_unload()` into an fp16 base. **Every hyperparameter has a written justification** in `HYPERPARAMETERS`, which the notebook prints as a table. |
| Evaluation | `evaluate.py` | Greedy decoding with an identical system prompt for base and fine-tuned models. ROUGE-L, plus task-level metrics (valid JSON, typology/action/risk accuracy), BERTScore F1, and an **LLM-as-judge** (gpt-oss-120b, 4-criterion rubric, Pydantic-validated JSON). Manual review sheet with automatic pre-labels (`outputs/manual_review.csv`). |
| Bonus RAG | `rag.py` | Answer perplexity is the confidence signal. The threshold is the validation-set p75. The ChromaDB store holds an 11-entry typology handbook plus **training** cases only (no leakage). Low-confidence answers are re-queried with the retrieved context. |

## Verification done before the Colab run
- `tests/test_data_pipeline.py`: 10 offline tests covering grounding, consistency, seeds, the exact 120/15/15 split, chat roles, parsing and metrics, review labels, near-duplicates, epoch-loss aggregation, and hyperparameter documentation.
- **CPU smoke test** of the real `train()` → `epoch_losses()` → `merge_and_save()` → `generate()` → metrics path, using Qwen2.5-0.5B for 2 steps. It found that **TRL 1.x defaults to `bf16=True`, which crashes on a T4**. `bf16=False` is now set explicitly and documented.

## Teacher system prompt
The full text is in `aml_triage/prompts.py::TEACHER_SYSTEM_PROMPT` and printed in the notebook (§2A.1 and Appendix).

## Run on Colab
1. Open the notebook with the badge, then *Runtime → T4 GPU*.
2. Add the secrets `GROQ_API_KEY`, `HF_TOKEN` (write access) and `HF_REPO_ID` (e.g. `yourname/qwen2.5-1.5b-aml-alert-triage`).
3. *Run all* (about 30–40 minutes). Then fill `reviewer_label` in `outputs/manual_review.csv`, re-run the two hallucination cells, and write the two qualitative paragraphs.
