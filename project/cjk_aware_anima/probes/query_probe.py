import torch, torch.nn.functional as F
from library.anima.weights import load_qwen3_text_encoder, load_qwen3_tokenizer, load_t5_tokenizer, load_llm_adapter
from library.anima.strategy import AnimaTokenizeStrategy, AnimaTextEncodingStrategy
torch.manual_seed(0)
QW='models/text_encoders/qwen_3_06b_base.safetensors'; DIT='models/diffusion_models/anima-base-v1.0.safetensors'
te,qtok=load_qwen3_text_encoder(QW,dtype=torch.float32,device='cpu'); t5=load_t5_tokenizer(None)
ad=load_llm_adapter(DIT,dtype=torch.float32,device='cpu')
tk=AnimaTokenizeStrategy(qtok,t5); es=AnimaTextEncodingStrategy()
UNK=t5.unk_token_id; EOS=t5.eos_token_id; PAD=t5.pad_token_id
@torch.no_grad()
def qwen_states(text):
    q_ids,q_mask,t_ids,t_mask=tk.tokenize([text]); pe,am,ti,tm=es.encode_tokens(tk,[te],[q_ids,q_mask,t_ids,t_mask]); return pe,am,ti,tm
@torch.no_grad()
def run(pe,am,ti,tm): return ad(source_hidden_states=pe,target_input_ids=ti,target_attention_mask=tm,source_attention_mask=am)[0]
def slotcos(a,b,n): return F.cosine_similarity(a[:n],b[:n],dim=-1)
prompts=["1girl, solo, hatsune miku, school uniform, cat ears, blonde hair, classroom, smile, looking at viewer",
         "1girl, solo, maid, holding a tray, cafe interior, masterpiece, best quality",
         "masterpiece, best quality, score_7, safe. An anime girl wearing a black tank-top is holding a sign that reads \"ANIMA\"."]
pe0,am0,ti0,tm0=qwen_states("")
for p in prompts:
    pe,am,ti,tm=qwen_states(p); n=int(tm.sum()); A=run(pe,am,ti,tm)
    E=ad.embed.weight[ti[0,:n]]                       # the T5 rows that entered
    B=run(pe0,am0,ti,tm)                              # Qwen side = "" (query only)
    tu=ti.clone(); tu[0,:n-1]=UNK; C=run(pe,am,tu,tm) # all query ids -> <unk> (the CJK-collapse case), eos kept
    perm=torch.randperm(n-1); ts=ti.clone(); ts[0,:n-1]=ti[0,perm]; D=run(pe,am,ts,tm)   # same rows, shuffled positions
    tr=ti.clone(); tr[0,:n-1]=torch.randint(3,32000,(n-1,)); Rr=run(pe,am,tr,tm)          # random rows
    toks=t5.convert_ids_to_tokens(ti[0,:n].tolist())
    print("\n== ",p[:70],"| slots",n)
    print("  out vs own T5 row (identity retained)   cos %.3f" % slotcos(A,E,n).mean())
    print("  out norm / row norm                     %.2f" % (A[:n].norm(dim=-1).mean()/E.norm(dim=-1).mean()))
    print("  Qwen side emptied  (query only)          cos %.3f  pooled %.3f" % (slotcos(A,B,n).mean(), F.cosine_similarity(A[:n].amax(0),B[:n].amax(0),dim=0)))
    print("  query -> all <unk> (Qwen intact)         cos %.3f  pooled %.3f" % (slotcos(A,C,n).mean(), F.cosine_similarity(A[:n].amax(0),C[:n].amax(0),dim=0)))
    print("  query rows shuffled (Qwen intact)        cos %.3f  (vs the row's own new position: %.3f)" % (slotcos(A,D,n).mean(), slotcos(A[perm],D[:n-1],n-1).mean()))
    print("  query rows random  (Qwen intact)         cos %.3f  pooled %.3f" % (slotcos(A,Rr,n).mean(), F.cosine_similarity(A[:n].amax(0),Rr[:n].amax(0),dim=0)))
    # per-slot: which slots keep identity most / least when Qwen emptied
    sc=slotcos(A,B,n); order=sc.argsort()
    print("  per-slot cos(query-only vs stock), lowest:", [(toks[i],round(float(sc[i]),2)) for i in order[:5].tolist()], " highest:", [(toks[i],round(float(sc[i]),2)) for i in order[-4:].tolist()])
    # slot-to-slot structure: are output slots distinct from each other?
    G=F.normalize(A[:n],dim=-1)@F.normalize(A[:n],dim=-1).T; off=(G.sum()-n)/(n*n-n)
    Gi=F.normalize(E,dim=-1)@F.normalize(E,dim=-1).T; offi=(Gi.sum()-n)/(n*n-n)
    print("  mean off-diag slot cos: rows-in %.3f -> out %.3f" % (offi, off))
