import torch, torch.nn.functional as F
from safetensors import safe_open
from library.anima import ext_vocab
from library.anima.weights import load_qwen3_tokenizer, load_t5_tokenizer
torch.manual_seed(0)
def rt(p,k):
    with safe_open(p,framework='pt') as f: return f.get_tensor(k)
t5e=rt('models/diffusion_models/anima-base-v1.0.safetensors','net.llm_adapter.embed.weight').float()
qe=rt('models/text_encoders/qwen_3_06b_base.safetensors','model.embed_tokens.weight').float()
qtok=load_qwen3_tokenizer('models/text_encoders/qwen_3_06b_base.safetensors'); t5=load_t5_tokenizer(None)
clean=ext_vocab.collect_clean_qwen_tokens(qtok); t5_ids,q_ids=ext_vocab.build_anchor_pairs(t5,qtok)
print(f"clean CJK qwen tokens {len(clean)}, anchors {len(t5_ids)}")
A=qe[q_ids]; B=t5e[t5_ids]; g=torch.Generator().manual_seed(0); perm=torch.randperm(len(t5_ids),generator=g); hold,train=perm[:1000],perm[1000:]
samp=torch.tensor(list(clean.keys()))[torch.randperm(len(clean))[:2000]]; X=qe[samp]
def pr(M):
    C=torch.cov((M-M.mean(0)).T); ev=torch.linalg.eigvalsh(C).clamp(min=0); return float(ev.sum()**2/(ev**2).sum())
def sep(M):
    Mn=F.normalize(M,dim=-1); G=Mn@Mn.T; n=len(M); off=G[~torch.eye(n,dtype=bool)]; return float(off.mean()), float((off>0.5).float().mean()*100)
def report(name,W,hc=None):
    Y=X@W; m,s=sep(Y); hcs=hc if hc is not None else float(F.cosine_similarity(A[hold]@W,B[hold],dim=-1).mean())
    print(f"{name:36s} holdout cos {hcs:.3f} | ext-key PR {pr(Y):6.1f} | pairwise cos {m:.3f}, >0.5: {s:4.1f}%")
m,s=sep(t5e[torch.randint(3,32100,(2000,))]); print(f"{'reference: native T5 rows':36s} {'':17s} | PR {pr(t5e[torch.randint(3,32100,(2000,))]):6.1f} | pairwise cos {m:.3f}, >0.5: {s:4.1f}%")
m,s=sep(X); print(f"{'reference: Qwen rows of ext tokens':36s} {'':17s} | PR {pr(X):6.1f} | pairwise cos {m:.3f}, >0.5: {s:4.1f}%")
At,Bt=A[train],B[train]; d=At.shape[1]
for ridge in (1e-2,1e-3,1e-4,1e-6):
    lam=ridge*At.pow(2).mean()*len(train); W=torch.linalg.solve(At.T@At+lam*torch.eye(d),At.T@Bt); report(f"ridge {ridge:g}"+(" (shipped)" if ridge==1e-2 else ""),W)
# orthogonal Procrustes (after centering + scaling): preserves the Qwen spectrum entirely
ma,mb=At.mean(0),Bt.mean(0); U,S,Vt=torch.linalg.svd((At-ma).T@(Bt-mb)); R=U@Vt; sc=float((Bt-mb).norm()/((At-ma)@R).norm())
Wp=R*sc; Yh=(A[hold]-ma)@Wp+mb; hc=float(F.cosine_similarity(Yh,B[hold],dim=-1).mean())
Y=(X-ma)@Wp+mb; m,s=sep(Y); print(f"{'orthogonal Procrustes (+mean)':36s} holdout cos {hc:.3f} | ext-key PR {pr(Y):6.1f} | pairwise cos {m:.3f}, >0.5: {s:4.1f}%")
# ridge but with the residual (unexplained) Qwen variance added back through Procrustes: 'ridge + orthogonal remainder'
lam=1e-2*At.pow(2).mean()*len(train); W=torch.linalg.solve(At.T@At+lam*torch.eye(d),At.T@Bt)
for alpha in (0.3,0.6):
    Wm=W+alpha*Wp; Yh=A[hold]@Wm; hc=float(F.cosine_similarity(Yh,B[hold],dim=-1).mean()); Y=X@Wm; m,s=sep(Y)
    print(f"{f'ridge + {alpha}·Procrustes':36s} holdout cos {hc:.3f} | ext-key PR {pr(Y):6.1f} | pairwise cos {m:.3f}, >0.5: {s:4.1f}%")
