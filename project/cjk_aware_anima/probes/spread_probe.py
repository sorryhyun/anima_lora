import torch, torch.nn.functional as F, json
from pathlib import Path
from library.anima.weights import load_qwen3_text_encoder, load_t5_tokenizer, load_llm_adapter
from library.anima.strategy import AnimaTokenizeStrategy, AnimaTextEncodingStrategy
from library.anima import ext_vocab
torch.manual_seed(0)
QW='models/text_encoders/qwen_3_06b_base.safetensors'; DIT='models/diffusion_models/anima-base-v1.0.safetensors'
te,qtok=load_qwen3_text_encoder(QW,dtype=torch.float32,device='cpu'); t5=load_t5_tokenizer(None)
ad=load_llm_adapter(DIT,dtype=torch.float32,device='cpu'); tk=AnimaTokenizeStrategy(qtok,t5); es=AnimaTextEncodingStrategy()
base=ad.embed.weight.data.clone(); N0=base.shape[0]
table,mapping=ext_vocab.load_ext_assets(Path('output/ckpt/cjk_vocab_pack_synthja_v5')); enc=ext_vocab.HybridT5Encoder.from_mapping(t5,qtok,mapping)
ad.embed=torch.nn.Embedding.from_pretrained(torch.cat([base,table.float()]))
q,qm,_,_=tk.tokenize([""]); pe0,am0,_,_=es.encode_tokens(tk,[te],[q,qm,torch.zeros(1,512,dtype=torch.long),torch.zeros(1,512,dtype=torch.long)])
L=64
@torch.no_grad()
def codes(ids):
    out=[]
    for s in range(0,len(ids),L-1):
        chunk=ids[s:s+L-1]+[t5.eos_token_id]; ti=torch.full((1,L),t5.pad_token_id); tm=torch.zeros(1,L,dtype=torch.long); ti[0,:len(chunk)]=torch.tensor(chunk); tm[0,:len(chunk)]=1
        o=ad(source_hidden_states=pe0,target_input_ids=ti,target_attention_mask=tm,source_attention_mask=am0)[0][:len(chunk)-1]; out.append(o)
    return torch.cat(out)
def report(name,X):
    Xn=F.normalize(X,dim=-1); G=Xn@Xn.T; n=len(X); off=G[~torch.eye(n,dtype=bool)]
    mu=Xn.mean(0).norm()
    C=torch.cov((X-X.mean(0)).T); ev=torch.linalg.eigvalsh(C).clamp(min=0); pr=float(ev.sum()**2/(ev**2).sum()); top=float(ev.flip(0)[:10].sum()/ev.sum())
    print(f"{name:34s} pairwise cos mean {off.mean():.3f} sd {off.std():.3f} | >0.5: {float((off>0.5).float().mean())*100:.1f}% | common-direction |mean| {mu:.2f} | eff. dim (PR) {pr:6.1f} | top-10 PCs {top*100:.0f}% var")
# samples
t5_ids=torch.randint(3,32100,(2000,)).tolist()
# frequent EN tag tokens: from a tag list
tags="1girl solo school uniform cat ears blonde hair classroom smile maid cafe long hair blush bedroom comic monochrome greyscale text sign sky tree outdoors indoors dress skirt shirt jacket glasses hat ribbon bow sword gun car window door table chair book flower water night day red blue green black white pink purple".split()
tag_ids=sorted({i for w in tags for i in t5(w,add_special_tokens=False)['input_ids']})
ext_rand=[N0+i for i in torch.randint(0,table.shape[0],(2000,)).tolist()]
recs=[json.loads(l)['text'] for l in open('post_image_dataset/cjk_unmask/ocr_records_sincos_ppocr.jsonl')]
ext_vis=sorted({i for l in recs for i in enc.encode(l,512)[0] if i>=N0})
print(f"samples: t5 random {len(t5_ids)}, EN tag pieces {len(tag_ids)}, ext random {len(ext_rand)}, ext visited-by-OCR {len(ext_vis)}")
print("-- raw embedding rows (the keys)")
report("T5 rows, random 2000", base[t5_ids]); report("ext rows (v5), random 2000", table[[i-N0 for i in ext_rand]].float()); report("ext rows (v5), OCR-visited", table[[i-N0 for i in ext_vis]].float())
print("-- adapter output codes (what the DiT sees), Qwen side empty")
report("T5 codes, random 2000", codes(t5_ids)); report("T5 codes, EN tag pieces", codes(tag_ids)); report("ext codes (v5), random 2000", codes(ext_rand)); report("ext codes (v5), OCR-visited", codes(ext_vis))
