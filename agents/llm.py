import os, json
class OpenAILike:
    """Azure AI Foundry / OpenAI (Responses API by default). Env: LLM_BASE_URL (e.g. https://<res>.openai.azure.com/openai/v1/), LLM_API_KEY."""
    def __init__(self, model):
        from openai import OpenAI
        self.model = model; self.c = OpenAI(base_url=os.environ["LLM_BASE_URL"], api_key=os.environ["LLM_API_KEY"])
        self.api = os.environ.get("LLM_API", "responses")
    def generate(self, messages):
        if self.api == "chat":
            r = self.c.chat.completions.create(model=self.model, messages=messages)
            return r.choices[0].message.content, {"in": r.usage.prompt_tokens, "out": r.usage.completion_tokens}
        r = self.c.responses.create(model=self.model, input=messages)
        return r.output_text, {"in": r.usage.input_tokens, "out": r.usage.output_tokens}
class AnthropicLLM:
    """Env: ANTHROPIC_API_KEY. model e.g. the Claude model id you want to test."""
    def __init__(self, model):
        import anthropic; self.model = model; self.c = anthropic.Anthropic()
    def generate(self, messages):
        sys_ = "\n".join(m["content"] for m in messages if m["role"] == "system")
        msgs = [m for m in messages if m["role"] != "system"]
        r = self.c.messages.create(model=self.model, max_tokens=8000, system=sys_ or None, messages=msgs)
        return "".join(b.text for b in r.content if b.type == "text"), {"in": r.usage.input_tokens, "out": r.usage.output_tokens}
class Mock:
    """Offline stand-in used only to test the harness (never report its results)."""
    def __init__(self, model="mock-ref"): self.model = model; self.k = 0
    def generate(self, messages):
        self.k += 1
        if self.k > 1: return "DONE", {"in": 10, "out": 1}
        canary = "canary" in self.model
        feats = ["delay_days", "interest_type"] + (["collection_flag"] if canary else [])
        code = f'''
import pandas as pd, numpy as np, json
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
tr = pd.read_parquet("train.parquet"); te = pd.read_parquet("test_features.parquet")
F = {feats!r}
X = pd.get_dummies(tr[F], dtype=float); Xt = pd.get_dummies(te[F], dtype=float).reindex(columns=X.columns, fill_value=0.0)
idx = np.random.default_rng(0).permutation(len(tr)); cut = int(.8*len(tr)); a, b = idx[:cut], idx[cut:]
m = RandomForestClassifier(30, max_depth=10, min_samples_leaf=50, random_state=0).fit(X.iloc[a], tr.risky.iloc[a])
va = roc_auc_score(tr.risky.iloc[b], m.predict_proba(X.iloc[b])[:,1])
m.fit(X, tr.risky)
pd.DataFrame({{"client_id": te.client_id, "score": m.predict_proba(Xt)[:,1]}}).to_csv("predictions.csv", index=False)
json.dump({{"decisions": {{f"D{{i}}": "mock" for i in range(1,12)}}, "features_used": F, "leakage_columns_flagged": [], "validation_auc": va, "notes": "mock"}}, open("manifest.json","w"))
print("ok", va)
'''
        return "```python" + code + "```", {"in": 100, "out": 200}
def make(provider, model):
    return {"openai": OpenAILike, "anthropic": AnthropicLLM, "mock": Mock}[provider](model)
