import torch, torch.nn.functional as F
from pathlib import Path
from library.anima.weights import load_qwen3_text_encoder, load_t5_tokenizer, load_llm_adapter
from library.anima.strategy import AnimaTokenizeStrategy, AnimaTextEncodingStrategy
from library.anima import ext_vocab
torch.manual_seed(0)
QW='models/text_encoders/qwen_3_06b_base.safetensors'; DIT='models/diffusion_models/anima-base-v1.0.safetensors'
te,qtok=load_qwen3_text_encoder(QW,dtype=torch.float32,device='cpu'); t5=load_t5_tokenizer(None)
ad=load_llm_adapter(DIT,dtype=torch.float32,device='cpu'); tk=AnimaTokenizeStrategy(qtok,t5); es=AnimaTextEncodingStrategy()
for p in ad.parameters(): p.requires_grad_(False)
base=ad.embed.weight.data.clone(); N0=base.shape[0]; L=64
cos=lambda a,b: F.cosine_similarity(a,b,dim=-1)
@torch.no_grad()
def qwen(text):
    q,qm,_,_=tk.tokenize([text]); pe,am,_,_=es.encode_tokens(tk,[te],[q,qm,torch.zeros(1,512,dtype=torch.long),torch.zeros(1,512,dtype=torch.long)]); return pe,am
def run_rows(pe,am,ids,row=None):
    ti=torch.full((1,L),t5.pad_token_id); tm=torch.zeros(1,L,dtype=torch.long); ti[0,:len(ids)]=torch.tensor(ids); tm[0,:len(ids)]=1
    x=base[ti[0]].clone()
    if row is not None: x=torch.where((ti[0]==-1)[:,None], row.expand(L,-1), x)
    ti2=ti.clone(); ti2[ti2==-1]=0
    # replicate adapter.forward with a custom input embedding
    h=ad.in_proj(x[None]); ctx=pe
    pos=torch.arange(L)[None]; posc=torch.arange(ctx.shape[1])[None]
    pe_=ad.rotary_emb(h,pos); pec=ad.rotary_emb(h,posc)
    for b in ad.blocks: h=b(h,ctx,target_attention_mask=tm.bool(),source_attention_mask=am.bool(),position_embeddings=pe_,position_embeddings_context=pec)
    return ad.norm(ad.out_proj(h))[0]
def t5ids(s): return t5(s,add_special_tokens=False)['input_ids']
PRE="1girl, solo, "; POST=", classroom, smile"
cases=[("ミク","miku"),("霊夢","reimu"),("アスカ","asuka"),("制服","school uniform"),("猫耳","cat ears"),("メイド","maid")]
print(f"{'JA':6s} {'EN':14s} | mean-EN-rows init -> after 150 steps of code-space inversion | per-slot cos to each EN piece")
for ja,en in cases:
    pre,post=t5ids(PRE),t5ids(POST); en_ids=t5ids(en); eos=[t5.eos_token_id]
    with torch.no_grad():
        peE,amE=qwen(PRE+en+POST); T=run_rows(peE,amE,pre+en_ids+post+eos)[len(pre):len(pre)+len(en_ids)]; Tm=T.mean(0)
    peJ,amJ=qwen(PRE+ja+POST)
    row=base[en_ids].mean(0).clone().requires_grad_(True); opt=torch.optim.Adam([row],lr=2e-2)
    ids=pre+[-1]+post+eos
    with torch.no_grad(): c0=float(cos(run_rows(peJ,amJ,ids,row)[len(pre)],Tm))
    for it in range(150):
        out=run_rows(peJ,amJ,ids,row)[len(pre)]; loss=1-cos(out,Tm); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad(): o=run_rows(peJ,amJ,ids,row)[len(pre)]; c1=float(cos(o,Tm)); per=[round(float(cos(o,T[k])),2) for k in range(len(en_ids))]
    print(f"{ja:6s} {en:14s} |  {c0:.2f} -> {c1:.2f}   | {per}  (row norm {float(row.norm()):.1f} vs base row norm {float(base[en_ids].norm(dim=-1).mean()):.1f})")
