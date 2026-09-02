import torch, torch.nn.functional as F
from pathlib import Path
from library.anima.weights import load_qwen3_text_encoder, load_qwen3_tokenizer, load_t5_tokenizer, load_llm_adapter
from library.anima.strategy import AnimaTokenizeStrategy, AnimaTextEncodingStrategy
from library.anima import ext_vocab
torch.manual_seed(0)
QW='models/text_encoders/qwen_3_06b_base.safetensors'; DIT='models/diffusion_models/anima-base-v1.0.safetensors'
te,qtok=load_qwen3_text_encoder(QW,dtype=torch.float32,device='cpu'); t5=load_t5_tokenizer(None)
ad=load_llm_adapter(DIT,dtype=torch.float32,device='cpu')
tk=AnimaTokenizeStrategy(qtok,t5); es=AnimaTextEncodingStrategy()
base_embed=ad.embed.weight.data.clone()
@torch.no_grad()
def qwen_states(text):
    q_ids,q_mask,t_ids,t_mask=tk.tokenize([text]); return es.encode_tokens(tk,[te],[q_ids,q_mask,t_ids,t_mask])
@torch.no_grad()
def run(pe,am,ti,tm): return ad(source_hidden_states=pe,target_input_ids=ti,target_attention_mask=tm,source_attention_mask=am)[0]
cos=lambda a,b: F.cosine_similarity(a,b,dim=-1)
pe0,am0,_,_=qwen_states("")
# 1. per-slot table
p="1girl, solo, hatsune miku, school uniform, cat ears, blonde hair, classroom, smile"
pe,am,ti,tm=qwen_states(p); n=int(tm.sum()); A=run(pe,am,ti,tm); B=run(pe0,am0,ti,tm)
tu=ti.clone(); tu[0,:n-1]=t5.unk_token_id; C=run(pe,am,tu,tm)
toks=t5.convert_ids_to_tokens(ti[0,:n].tolist())
print("== per slot: cos(stock, Qwen-emptied) | cos(stock, query=<unk>)")
print("  "+"  ".join(f"{t}:{float(cos(A[i],B[i])):.2f}/{float(cos(A[i],C[i])):.2f}" for i,t in enumerate(toks)))
# 2. same row across prompts: is '▁cat' / 'ku' the same vector in another context?
p2="cat ears, outdoors, 2girls, hatsune miku, summer, eating ice cream"
pe2,am2,ti2,tm2=qwen_states(p2); n2=int(tm2.sum()); A2=run(pe2,am2,ti2,tm2); toks2=t5.convert_ids_to_tokens(ti2[0,:n2].tolist())
for w in ['▁cat','▁ears','ku','▁mi','hat',',']:
    i=toks.index(w); j=toks2.index(w); print(f"  row {w!r} across prompts: cos {float(cos(A[i],A2[j])):.2f}")
i=toks.index('▁cat'); print("  cross-row baseline (▁cat vs ▁hair):", round(float(cos(A[i],A[toks.index('▁hair')])),2))
# 3. JA line through the two packs
line="1girl, solo, @sincos, japanese text, 「あーもううっさい早く終わりなさいよ」, 猫耳, 制服"
res={}
for name in ('synthjako2','synthja_v5'):
    table,mapping=ext_vocab.load_ext_assets(Path(f'output/ckpt/cjk_vocab_pack_{name}'))
    enc=ext_vocab.HybridT5Encoder.from_mapping(t5,qtok,mapping)
    ad.embed=torch.nn.Embedding.from_pretrained(torch.cat([base_embed,table.float()]))
    ids,mask=enc.encode(line,512); ti_=torch.tensor([ids]); tm_=torch.tensor([mask])
    q_ids,q_mask,_,_=tk.tokenize([line]); pe_,am_,_,_=es.encode_tokens(tk,[te],[q_ids,q_mask,ti_,tm_])
    nj=int(tm_.sum()); res[name]=(run(pe_,am_,ti_,tm_)[:nj], run(pe0,am0,ti_,tm_)[:nj], ids[:nj])
    A_,B_,_=res[name]; ext=[k for k in range(nj) if ids[k]>=ext_vocab.T5_TABLE_SIZE]
    print(f"== {name}: slots {nj}, ext slots {len(ext)} | Qwen-emptied cos: EN slots {float(cos(A_,B_)[[k for k in range(nj) if k not in ext]].mean()):.2f}, ext slots {float(cos(A_,B_)[ext].mean()):.2f}")
A1,_,ids1=res['synthjako2']; A2_,_,ids2=res['synthja_v5']; assert ids1==ids2
ext=[k for k in range(len(ids1)) if ids1[k]>=ext_vocab.T5_TABLE_SIZE]; en=[k for k in range(len(ids1)) if k not in ext]
c=cos(A1,A2_); print(f"== jako2 vs v5 on the same line: cos EN slots {float(c[en].mean()):.3f} (min {float(c[en].min()):.3f}), ext slots {float(c[ext].mean()):.3f} (min {float(c[ext].min()):.3f}); pooled {float(cos(A1.amax(0),A2_.amax(0))):.3f}")
