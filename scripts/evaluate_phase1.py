#!/usr/bin/env python3
"""Local evaluator for Phase 1.

Usage:
  python evaluate_phase1.py --truth truth.json --prediction prediction.json

Requires: bert-score and torch for the official primary metric.
"""
import argparse, json, math, sys
from collections import Counter

try:
    import bert_score
except ImportError as exc:
    raise SystemExit("Missing dependency: bert-score. Install it from requirements.docx / requirements.txt.") from exc

class DuplicateJSONKeyError(ValueError): pass

def reject_constant(v): raise ValueError(f"Non-standard JSON numeric value '{v}' is not allowed.")

def pairs_no_duplicates(pairs):
    out = {}; dup = []
    for k, v in pairs:
        if k in out: dup.append(k)
        out[k] = v
    if dup: raise DuplicateJSONKeyError(f"Duplicate JSON object keys detected: {list(dict.fromkeys(dup))[:10]}.")
    return out

def load_json(path):
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f, object_pairs_hook=pairs_no_duplicates, parse_constant=reject_constant)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {path} at line {e.lineno}, column {e.colno}: {e.msg}") from e

def validate(data, name):
    if not isinstance(data, dict) or not data: raise ValueError(f"{name} must be a non-empty JSON object mapping IDs to records.")
    for i, rec in data.items():
        if not isinstance(i, str) or not i.strip() or not isinstance(rec, dict) or 'text' not in rec or not isinstance(rec['text'], str) or not rec['text'].strip():
            raise ValueError(f"Invalid record for ID {i!r} in {name}; each record must contain non-empty string field 'text'.")

def lcs_length(a,b):
    if not a or not b: return 0
    if len(a) < len(b): a,b=b,a
    masks={}
    for j,t in enumerate(a): masks[t]=masks.get(t,0)|(1<<j)
    state=0
    for t in b:
        m=masks.get(t,0); x=m|state; y=(state<<1)|1; state=x & ~(x-y)
    return state.bit_count() if hasattr(int,'bit_count') else bin(state).count('1')

def rouge_l_f1(p,r):
    pt=p.split(); rt=r.split()
    if not pt or not rt: return 0.0
    l=lcs_length(pt,rt)
    if not l: return 0.0
    pp=l/len(pt); rr=l/len(rt)
    return 0.0 if pp+rr==0 else 2*pp*rr/(pp+rr)

def ngrams(t,n): return Counter(tuple(t[i:i+n]) for i in range(len(t)-n+1)) if len(t)>=n else Counter()

def bleu4(preds, refs):
    order=4; clipped=[0]*order; totals=[0]*order; pl=rl=0
    for p,r in zip(preds,refs):
        pl+=len(p); rl+=len(r)
        for n in range(1,order+1):
            pn=ngrams(p,n); rn=ngrams(r,n); totals[n-1]+=sum(pn.values())
            for g,c in pn.items(): clipped[n-1]+=min(c,rn.get(g,0))
    prec=[clipped[i]/totals[i] if totals[i] else 0.0 for i in range(order)]
    if any(x<=0 for x in prec) or pl==0: return 0.0
    bp=1.0 if pl>rl else math.exp(1.0-rl/pl)
    return bp*math.exp(sum(math.log(x) for x in prec)/order)

def main():
    ap=argparse.ArgumentParser(description='Evaluate Phase 1 predictions on your own held-out split.')
    ap.add_argument('--truth', required=True, help='Your held-out reference JSON.')
    ap.add_argument('--prediction', required=True, help='Your prediction JSON.')
    args=ap.parse_args()
    truth=load_json(args.truth); pred=load_json(args.prediction)
    validate(truth,'truth'); validate(pred,'prediction')
    if set(truth)!=set(pred): raise ValueError(f"ID mismatch: missing={len(set(truth)-set(pred))}, extra={len(set(pred)-set(truth))}")
    ps=[pred[i]['text'].split() for i in truth]; rs=[truth[i]['text'].split() for i in truth]
    pred_texts=[pred[i]['text'].strip() for i in truth]; ref_texts=[truth[i]['text'].strip() for i in truth]
    try:
        import torch
        device='cuda' if torch.cuda.is_available() else 'cpu'
    except ImportError: device=None
    _,_,f=bert_score.score(pred_texts, ref_texts, lang='en', batch_size=8, verbose=False, **({'device':device} if device else {}))
    bertscore=float(f.mean().item())
    rouge=sum(rouge_l_f1(pred[i]['text'],truth[i]['text']) for i in truth)/len(truth)
    b4=bleu4(ps,rs)
    print(f'Examples: {len(truth)}')
    print(f'BERTScore F1: {bertscore:.6f}')
    print(f'ROUGE-L F1:   {rouge:.6f}')
    print(f'BLEU-4:       {b4:.6f}')
if __name__=='__main__':
    try: main()
    except Exception as e: print(f'EVALUATION FAILED: {e}', file=sys.stderr); sys.exit(1)
