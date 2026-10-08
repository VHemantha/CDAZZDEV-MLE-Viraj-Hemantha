"""Generates notebooks/Task2_AML_Triage_QLoRA.ipynb -- run it on a Colab T4 GPU."""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Generate the Task 2 Colab notebook: dataset
# engineering, QLoRA training, evaluation, RAG fallback', Date: 2026-10-07

from pathlib import Path

import nbformat as nbf

REPO = "https://github.com/YOUR_GITHUB_USERNAME/CDAZZDEV-MLE-YourName"
COLAB = ("https://colab.research.google.com/github/YOUR_GITHUB_USERNAME/CDAZZDEV-MLE-YourName/"
         "blob/main/task2_genai/notebooks/Task2_AML_Triage_QLoRA.ipynb")
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

cells = [
    md(f"""# Task 2: Domain-Specific Fine-Tuning · AML Transaction-Alert Triage
**CDAZZDEV Senior MLE Assessment · Generative AI** · QLoRA (4-bit NF4) on `Qwen/Qwen2.5-1.5B-Instruct` · Colab T4

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({COLAB})

> **Runtime → Change runtime type → T4 GPU.** Add the Colab secrets `GROQ_API_KEY` (LLM judge) and `HF_TOKEN` (write access, for pushing the model).

## Problem statement
Bank transaction-monitoring systems raise thousands of alerts a day, and most of them are false positives. A level-1 analyst triages each alert: identify the typology, weigh the red flags against mitigating facts, and choose an action. General-purpose LLMs do this badly. They ignore the bank's label vocabulary, escalate everything, and, most dangerously, **invent amounts or counterparties**, which is unacceptable in a regulatory filing.

| | Specification |
|---|---|
| **Input** (user turn) | A rendered alert: rule triggered; customer profile (segment, business, tenure, KYC rating, declared turnover, residence); prior alerts; a table of 3–10 transactions (date, direction, type, amount, currency, counterparty, country); optional analyst context |
| **Output** (assistant turn) | JSON with `typology` (one of 11), `risk_level` (low/medium/high), `red_flags[]`, `mitigating_factors[]`, `recommended_action` (close_no_action / request_information / enhanced_due_diligence / escalate_sar), and a 2–3 sentence `rationale` |
| **Correct** | Valid JSON using only the allowed labels; typology and action match the investigator's gold decision; every cited fact (amount, date, counterparty, country) appears in the alert |
| **Partially correct** | Valid and grounded, but typology *or* action differs from gold (e.g. EDD instead of SAR) |
| **Incorrect / hallucinated** | Invalid JSON or labels; or any fact not present in the alert (a fabricated amount, total, document or counterparty) |"""),
    code(f"""import os, sys, subprocess, json, gc, time
from pathlib import Path
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    if not Path("/content/repo").exists():
        subprocess.run(["git", "clone", "-q", "{REPO}", "/content/repo"], check=True)
    os.chdir("/content/repo/task2_genai")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], check=True)
    from google.colab import userdata
    for k in ("GROQ_API_KEY", "HF_TOKEN", "HF_REPO_ID"):
        try: os.environ[k] = userdata.get(k)
        except Exception: pass
else:
    if Path.cwd().name == "notebooks": os.chdir(Path.cwd().parent)
    from dotenv import load_dotenv; load_dotenv(Path.cwd().parent / ".env", override=True)
sys.path.insert(0, str(Path.cwd()))
print({{k: ("set" if os.getenv(k) else "missing") for k in ("GROQ_API_KEY", "HF_TOKEN")}})
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"], capture_output=True, text=True).stdout)"""),
    code("""import numpy as np, pandas as pd, matplotlib.pyplot as plt
from IPython.display import display, Markdown
from aml_triage import config
from aml_triage.prompts import TEACHER_SYSTEM_PROMPT, STUDENT_SYSTEM_PROMPT
from aml_triage.schema import render_alert, render_triage
from aml_triage import dataset as D
pd.set_option("display.max_colwidth", 160)
config.OUTPUT_DIR.mkdir(exist_ok=True)"""),

    md("""---
# 2A: Dataset engineering
## 2A.1 Teacher generation
* **Teacher:** `openai/gpt-oss-120b` via Groq (free tier), with OpenRouter and Qwen3.8-27B as fallbacks when a daily quota runs out. The teacher used is recorded per case. **The student, Qwen2.5-1.5B, is a different model.**
* **Diversity by construction:** each case comes from a *seed* that fixes the true typology (balanced, with 22% benign false positives), customer segment (15), jurisdictions (20), difficulty (`clear-cut` / `borderline` / `misleading-surface`) and transaction count.
* **Quality gate:** every case is validated with Pydantic and then with consistency checks: the label matches the seed, risk and action are coherent, and a **grounding check** rejects any figure the alert doesn't support (it handles totals, `1.07M`-style abbreviations and regulatory thresholds). Rejected cases are regenerated.

150 seeds were drawn; **131 cases passed the quality gate** within the free-tier token budget (112 from gpt-oss-120b, 19 from qwen3.8-27b). The dataset was generated once with `scripts/generate_dataset.py` and committed, so this notebook reuses it. Set `REGENERATE=True` to rebuild it (this costs about 150K teacher tokens). The full teacher system prompt is printed here and in the appendix."""),
    code("""REGENERATE = False
if REGENERATE or not config.RAW_PATH.exists():
    from aml_triage.generate import generate
    generate()
cases = D.load_cases()
print(f"{len(cases)} validated cases in {config.RAW_PATH.name}\\n")
print("=== TEACHER SYSTEM PROMPT (full) ===\\n" + TEACHER_SYSTEM_PROMPT)"""),
    code("""ex = cases[0]
print("Seed:", ex.seed, "| teacher:", ex.teacher, "\\n")
print("USER TURN (model input):\\n" + render_alert(ex.alert))
print("\\nASSISTANT TURN (target):\\n" + render_triage(ex.triage))"""),
    md("## 2A.2 Diversity validation"),
    code("""rep = D.diversity_report(cases)
print("Prompt length (words):", rep["prompt_words"], "\\nAnswer length (words):", rep["answer_words"])
print(f"distinct-1 {rep['distinct_1']:.3f} | distinct-2 {rep['distinct_2']:.3f} | rules {rep['n_distinct_rules']} | "
      f"countries {rep['n_distinct_counterparty_countries']} | currencies {rep['n_distinct_currencies']}")
nd = {k: v for k, v in rep["near_duplicates"].items() if not k.startswith("_")}
print("Near-duplicate analysis (5-gram shingle Jaccard over all pairs):", {k: round(v, 3) if isinstance(v, float) else v for k, v in nd.items()})
print("Teachers:", dict(rep["teacher"]))

fig, ax = plt.subplots(2, 3, figsize=(16, 8.5))
ax[0, 0].hist(rep["_prompt_words"], bins=20, color="#2a78d6"); ax[0, 0].set_title("Prompt length (words)")
ax[0, 1].hist(rep["near_duplicates"]["_all"], bins=40, color="#4a3aa7"); ax[0, 1].set_title("Pairwise 5-gram Jaccard (all pairs)")
ax[0, 1].axvline(0.5, color="#e34948", ls="--"); ax[0, 1].text(0.51, ax[0, 1].get_ylim()[1] * 0.8, "near-dup\\nthreshold", color="#e34948")
for a, key in [(ax[0, 2], "typology"), (ax[1, 0], "customer_segment"), (ax[1, 1], "recommended_action")]:
    s = pd.Series(rep[key]).sort_values(); a.barh(s.index, s.values, color="#1baf7a"); a.set_title(key)
kw = pd.Series(dict(rep["top_keywords"])).sort_values()
ax[1, 2].barh(kw.index[-20:], kw.values[-20:], color="#eb6834"); ax[1, 2].set_title("Top keywords (rules, red flags, rationales)")
for a in ax.flat: a.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig(config.OUTPUT_DIR / "dataset_diversity.png", dpi=110); plt.show()
display(pd.DataFrame({"difficulty": pd.Series(rep["difficulty"]), }).T, pd.DataFrame({"risk_level": pd.Series(rep["risk_level"])}).T)"""),
    md("## 2A.3 JSONL chat format and a stratified 80/10/10 split"),
    code("""splits = D.stratified_split(cases)
sizes = D.write_splits(splits)
print("Split sizes:", sizes, "| total", sum(sizes.values()))
display(pd.DataFrame({s: pd.Series([c.triage.typology for c in splits[s]]).value_counts() for s in splits}).fillna(0).astype(int))
train_rows, val_rows, test_rows = (D.read_split(s) for s in ("train", "val", "test"))
print("\\nOne JSONL record (conversational prompt/completion format):")
print(json.dumps(train_rows[0], indent=1, ensure_ascii=False)[:1500], "...")"""),
    code("""from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(config.BASE_MODEL)
rendered = tok.apply_chat_template(train_rows[0]["prompt"] + train_rows[0]["completion"], tokenize=False)
print("Rendered with the base model's own chat template (Qwen2.5 ChatML):\\n")
print(rendered[:900], "\\n...\\n", rendered[-300:])
lens = [len(tok.apply_chat_template(r["prompt"] + r["completion"], tokenize=True)) for r in train_rows + val_rows + test_rows]
print(f"\\nTokens per example: mean {np.mean(lens):.0f}, max {max(lens)} -> max_seq_length=1024 truncates nothing: {max(lens) < 1024}")"""),

    md("""---
# 2B: QLoRA fine-tuning
## 2B.1 Every hyperparameter and why
This table is generated from `aml_triage/train.py::HYPERPARAMETERS`, the same dict the configs are built from, so the documentation can't drift from what was trained."""),
    code("""from aml_triage.train import hyperparameter_table, HP
display(pd.DataFrame(hyperparameter_table()))"""),
    md("""## 2B.2 Train (4-bit NF4 base + LoRA adapters), logging train and val loss
**OOM handling:** if the T4 runs out of memory, the cell records the error, halves the micro-batch, doubles gradient accumulation (so the effective batch stays at 8), and retries. The log below shows what happened in this run."""),
    code("""import torch
from aml_triage import train as TR
from peft import PeftModel
oom_log = []
attempt_cfg = [(HP["per_device_train_batch_size"], HP["gradient_accumulation_steps"]), (1, 8)]
for bs, ga in attempt_cfg:
    TR.HP["per_device_train_batch_size"], TR.HP["gradient_accumulation_steps"] = bs, ga
    try:
        t0 = time.time()
        trainer = TR.train(train_rows, val_rows, output_dir="outputs/qlora_run")
        print(f"Training finished in {(time.time() - t0) / 60:.1f} min with batch {bs} x accum {ga}")
        break
    except torch.cuda.OutOfMemoryError as e:
        oom_log.append({"batch_size": bs, "grad_accum": ga, "error": str(e)[:200]})
        print(f"OOM with batch {bs} -> retrying with smaller micro-batch (effective batch unchanged)")
        gc.collect(); torch.cuda.empty_cache()
print("OOM events:", oom_log or "none")
trainer.model.print_trainable_parameters()
print("Quantisation:", trainer.model.base_model.model.config.quantization_config)"""),
    code("""ep = pd.DataFrame(TR.epoch_losses(trainer.state.log_history))
display(ep.round(4))
steps = pd.DataFrame([r for r in trainer.state.log_history if "loss" in r])
fig, a = plt.subplots(figsize=(8, 4))
a.plot(steps["epoch"], steps["loss"], color="#2a78d6", lw=1.5, label="train loss (per 5 steps)")
a.plot(ep["epoch"], ep["val_loss"], "o-", color="#eb6834", lw=2, label="validation loss (per epoch)")
a.set_xlabel("epoch"); a.set_ylabel("loss"); a.legend(frameon=False); a.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig(config.OUTPUT_DIR / "loss_curves.png", dpi=110); plt.show()
dec = all(b < a_ for a_, b in zip(ep["val_loss"], ep["val_loss"][1:]))
print("Validation loss strictly decreasing across epochs:", dec)
ep.to_csv(config.OUTPUT_DIR / "epoch_losses.csv", index=False)"""),
    md("## 2B.3 Merge the adapters with `merge_and_unload()`, save, and push to the Hugging Face Hub"),
    code("""ADAPTER_DIR, MERGED_DIR = "outputs/adapter", "outputs/merged"
trainer.save_model(ADAPTER_DIR)
del trainer; gc.collect(); torch.cuda.empty_cache()
ft_model, ft_tok = TR.merge_and_save(ADAPTER_DIR, MERGED_DIR)
print("Merged model saved:", sorted(os.listdir(MERGED_DIR)))
repo_id = os.getenv("HF_REPO_ID") or config.HF_REPO_ID
if os.getenv("HF_TOKEN") and "YOUR_HF_USERNAME" not in repo_id:
    ft_model.push_to_hub(repo_id, token=os.environ["HF_TOKEN"], private=False)
    ft_tok.push_to_hub(repo_id, token=os.environ["HF_TOKEN"])
    print("Pushed: https://huggingface.co/" + repo_id)
else:
    print("Set HF_TOKEN and HF_REPO_ID secrets to push; model kept at", MERGED_DIR)"""),

    md("""---
# 2C: Evaluation against the base model (same system prompt, same held-out test set)
Both models run in fp16 with greedy decoding and an identical system prompt. The only difference is the fine-tuning."""),
    code("""from aml_triage import evaluate as E
from transformers import AutoModelForCausalLM
ft_model.eval()
ft_preds = E.run_inference(ft_model, ft_tok, test_rows, "fine-tuned")
base_model = AutoModelForCausalLM.from_pretrained(config.BASE_MODEL, device_map="auto", **TR._dtype_kw(torch.float16)).eval()
base_preds = E.run_inference(base_model, ft_tok, test_rows, "base")
del base_model; gc.collect(); torch.cuda.empty_cache()
golds = [r["completion"][0]["content"] for r in test_rows]
json.dump({"base": base_preds, "fine_tuned": ft_preds}, open(config.OUTPUT_DIR / "test_predictions.json", "w"), indent=1)"""),
    md("## 2C.1 ROUGE-L and task-level accuracy"),
    code("""rows = []
for name, preds in [("Base (system prompt only)", base_preds), ("Fine-tuned (QLoRA)", ft_preds)]:
    texts = [p["text"] for p in preds]
    rl = E.rouge_l(texts, golds)
    fm = E.field_metrics(texts, golds)
    rows.append({"model": name, "ROUGE-L (mean)": np.mean(rl), "ROUGE-L (median)": np.median(rl), **fm})
results = pd.DataFrame(rows).set_index("model")
display(results.round(3))
results.to_csv(config.OUTPUT_DIR / "metrics_rouge_fields.csv")"""),
    md("## 2C.2 Additional metrics: BERTScore F1 and an LLM-as-judge with a structured JSON rubric"),
    code("""from aml_triage.schema import Case
test_cases = {c.case_id: c for c in cases}
judge_rows = []
for name, preds in [("base", base_preds), ("fine-tuned", ft_preds)]:
    f1 = E.bertscore_f1([p["text"] for p in preds], golds)
    for p, g, r, f in zip(preds, golds, test_rows, f1):
        js = E.judge(r["prompt"][1]["content"], g, p["text"])
        judge_rows.append({"model": name, "case_id": p["case_id"], "bertscore_f1": f,
                           **(js.model_dump() if js else {})})
J = pd.DataFrame(judge_rows)
summary = J.groupby("model")[["bertscore_f1", "decision_accuracy", "grounding", "reasoning_quality",
                              "format_compliance", "hallucination"]].mean().round(3)
display(summary)
print("Example judge output (structured JSON):"); print(J.iloc[-1][["decision_accuracy", "grounding", "reasoning_quality", "format_compliance", "hallucination", "comment"]].to_dict())
J.to_csv(config.OUTPUT_DIR / "judge_scores.csv", index=False)"""),
    md("""## 2C.3 Hallucination rate: manual review of fine-tuned outputs
All 13 held-out test outputs (the brief asks for at least 10) are listed below with an **automatic pre-label** to speed up review. The pre-label combines invented-figure detection, the judge's hallucination flag, and label agreement. **The final labels in `reviewer_label` are assigned by a human reviewer**, who reads each alert next to the output; the pre-label is only a suggestion. The sheet is saved to `outputs/manual_review.csv`."""),
    code("""review = []
jl = J[J.model == "fine-tuned"].set_index("case_id")
for p, g, r in zip(ft_preds, golds, test_rows):
    case = test_cases[p["case_id"]]
    js = jl.loc[p["case_id"]] if p["case_id"] in jl.index else None
    judge_obj = E.JudgeScore(**js[["decision_accuracy", "grounding", "reasoning_quality", "format_compliance", "hallucination", "comment"]].to_dict()) if js is not None and not js.isna().any() else None
    label, why = E.suggest_label(p["text"], g, case.alert, judge_obj)
    pt, gt = E.parse_triage(p["text"]), E.parse_triage(g)
    review.append({"case_id": p["case_id"], "gold": f"{gt.typology} / {gt.recommended_action}",
                   "prediction": f"{pt.typology} / {pt.recommended_action}" if pt else "INVALID",
                   "suggested_label": label, "evidence": why, "reviewer_label": ""})
R = pd.DataFrame(review)
REVIEW_PATH = config.OUTPUT_DIR / "manual_review.csv"
if REVIEW_PATH.exists():          # keep the human's labels across re-runs
    prev = pd.read_csv(REVIEW_PATH).set_index("case_id")["reviewer_label"].fillna("")
    R["reviewer_label"] = R["case_id"].map(prev).fillna("")
R.to_csv(REVIEW_PATH, index=False)
display(R)"""),
    code("""labels = [h if h in E.REVIEW_LABELS else s for h, s in zip(R["reviewer_label"], R["suggested_label"])]
n_human = sum(h in E.REVIEW_LABELS for h in R["reviewer_label"])
print(f"Reviewed responses: {len(labels)} ({n_human} human-confirmed labels, {len(labels) - n_human} pre-labels pending review)")
print(pd.Series(labels).value_counts().to_string())
print(f"\\nHallucination rate (fine-tuned): {E.hallucination_rate(labels):.1f}%")"""),
    md("## 2C.4 Evidence for the qualitative analysis"),
    code("""bp = {p["case_id"]: E.parse_triage(p["text"]) for p in base_preds}
fp = {p["case_id"]: E.parse_triage(p["text"]) for p in ft_preds}
gp = {r["case_id"]: E.parse_triage(r["completion"][0]["content"]) for r in test_rows}
def lab(t): return f"{t.typology} / {t.recommended_action}" if t else "INVALID JSON"
cmp = pd.DataFrame([{"case_id": k, "difficulty": test_cases[k].seed["difficulty"], "gold": lab(gp[k]),
                     "base": lab(bp[k]), "fine-tuned": lab(fp[k]),
                     "base_ok": bool(bp[k] and bp[k].recommended_action == gp[k].recommended_action),
                     "ft_ok": bool(fp[k] and fp[k].recommended_action == gp[k].recommended_action)} for k in gp])
display(cmp)
fixed = cmp[(~cmp.base_ok) & cmp.ft_ok]
if len(fixed):
    k = fixed.iloc[0]["case_id"]
    print(f"\\nExample fixed by fine-tuning ({k}):\\n--- BASE ---")
    print(next(p["text"] for p in base_preds if p["case_id"] == k)[:900])
    print("--- FINE-TUNED ---"); print(next(p["text"] for p in ft_preds if p["case_id"] == k)[:900])"""),
    md("""### Qualitative analysis
**Where fine-tuning improved behaviour.** *[Write after the run, using the tables above: cite specific case_ids. For example, base-model outputs that broke the label vocabulary or the JSON contract and were fixed; cases where the base model escalated benign `misleading-surface` alerts that the fine-tuned model correctly closed; changes in the judge's grounding score.]*

**Remaining failure modes and next steps.** *[Write after the run: which typologies or difficulties still fail (see `cmp`), any ungrounded figures, and what would address them. For example: more `borderline` and `misleading-surface` examples for the confused typology pairs, DPO on (grounded, hallucinated) answer pairs, or larger r / more epochs if validation loss was still falling.]*"""),

    md("""---
# Bonus: RAG fallback layer (perplexity-gated, ChromaDB)
* **Confidence signal:** the perplexity of the generated answer, `exp(-mean token log-prob)`. A low-confidence answer has high perplexity.
* **Threshold:** the 75th percentile of perplexity on the **validation** set, so it's calibrated without touching the test set.
* **Knowledge base (ChromaDB):** the 11-entry AML typology handbook (`data/knowledge_base/`) plus the **training** cases with their gold decisions. Validation and test cases are excluded, so nothing leaks.
* **Fallback:** if perplexity is above the threshold, retrieve the top-2 handbook entries and top-2 similar cases, then re-query the model."""),
    code("""from aml_triage import rag
col = rag.build_store(train_rows)
val_ppl = [E.generate(ft_model, ft_tok, r["prompt"])["perplexity"] for r in val_rows]
thr = rag.calibrate_threshold(val_ppl)
print(f"Validation perplexity: median {np.median(val_ppl):.3f}, p75 threshold {thr:.3f}; KB size {col.count()} documents")
rag_out = [rag.answer_with_fallback(ft_model, ft_tok, col, r, thr) for r in test_rows]
rag_tab = pd.DataFrame([{"case_id": o["case_id"], "ppl": round(o["first"]["perplexity"], 3), "used_rag": o["used_rag"],
                         "gold": lab(gp[o["case_id"]]), "before": lab(E.parse_triage(o["first"]["text"])),
                         "after": lab(E.parse_triage(o["final"]["text"]))} for o in rag_out])
display(rag_tab)
acc = lambda col_: np.mean([E.parse_triage(o[col_]["text"]) is not None and E.parse_triage(o[col_]["text"]).recommended_action == gp[o["case_id"]].recommended_action for o in rag_out])
print(f"Action accuracy without RAG {acc('first'):.2f} -> with perplexity-gated RAG {acc('final'):.2f}")"""),
    code("""used = [o for o in rag_out if o["used_rag"]]
if used:
    o = max(used, key=lambda x: x["first"]["perplexity"])
    print(f"BEFORE/AFTER example ({o['case_id']}, perplexity {o['first']['perplexity']:.3f} > {thr:.3f})\\n")
    print("GOLD:", lab(gp[o["case_id"]]))
    print("\\n--- BEFORE (no retrieval) ---\\n" + o["first"]["text"][:900])
    print("\\n--- RETRIEVED CONTEXT (truncated) ---\\n" + o["context"][:700])
    print("\\n--- AFTER (with retrieval) ---\\n" + o["final"]["text"][:900])
else:
    print("No test answer exceeded the confidence threshold.")"""),
    md("---\n## Appendix: prompts used"),
    code("""print("=== TEACHER (data generation) ===\\n" + TEACHER_SYSTEM_PROMPT)
print("\\n=== STUDENT (base and fine-tuned, identical) ===\\n" + STUDENT_SYSTEM_PROMPT)
from aml_triage.prompts import JUDGE_SYSTEM_PROMPT; print("\\n=== LLM JUDGE ===\\n" + JUDGE_SYSTEM_PROMPT)"""),
]

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
    "language_info": {"name": "python"}, "accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []}})
out = Path(__file__).resolve().parents[1] / "notebooks" / "Task2_AML_Triage_QLoRA.ipynb"
nbf.write(nb, out)
print("wrote", out)
