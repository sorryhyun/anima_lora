import torch, torch.nn.functional as F
from pathlib import Path
from library.anima.weights import load_qwen3_text_encoder, load_t5_tokenizer, load_llm_adapter
from library.anima.strategy import AnimaTokenizeStrategy, AnimaTextEncodingStrategy
from library.anima import ext_vocab
QW='models/text_encoders/qwen_3_06b_base.safetensors'; DIT='models/diffusion_models/anima-base-v1.0.safetensors'
te,qtok=load_qwen3_text_encoder(QW,dtype=torch.float32,device='cpu'); t5=load_t5_tokenizer(None)
ad=load_llm_adapter(DIT,dtype=torch.float32,device='cpu'); tk=AnimaTokenizeStrategy(qtok,t5); es=AnimaTextEncodingStrategy()
base=ad.embed.weight.data.clone(); N0=base.shape[0]
table,mapping=ext_vocab.load_ext_assets(Path('output/ckpt/cjk_vocab_pack_synthja_v5')); enc=ext_vocab.HybridT5Encoder.from_mapping(t5,qtok,mapping)
cos=lambda a,b: F.cosine_similarity(a,b,dim=-1)
@torch.no_grad()
def qwen(text):
    q,qm,_,_=tk.tokenize([text]); pe,am,_,_=es.encode_tokens(tk,[te],[q,qm,torch.zeros(1,512,dtype=torch.long),torch.zeros(1,512,dtype=torch.long)]); return pe,am
@torch.no_grad()
def run(pe,am,ids,extra_rows=None):
    emb=base if extra_rows is None else torch.cat([base,table.float(),extra_rows])
    ad.embed=torch.nn.Embedding.from_pretrained(emb if extra_rows is not None else torch.cat([base,table.float()]))
    ti=torch.full((1,512),t5.pad_token_id); tm=torch.zeros(1,512,dtype=torch.long); ti[0,:len(ids)]=torch.tensor(ids); tm[0,:len(ids)]=1
    return ad(source_hidden_states=pe,target_input_ids=ti,target_attention_mask=tm,source_attention_mask=am)[0]
def t5ids(s): return t5(s,add_special_tokens=False)['input_ids']
PRE="1girl, solo, "; POST=", classroom, smile"
cases=[("ミク","miku"),("霊夢","reimu"),("アスカ","asuka"),("制服","school uniform"),("猫耳","cat ears"),("金髪","blonde hair"),("メイド","maid"),("笑顔","smile")]
print(f"{'JA':6s} {'EN':14s} | EN-ids on T5 (word_sub ceiling) | init: mean EN rows | init: mean ext char rows (current mint) | per-char ext rows, no mint")
for ja,en in cases:
    pre,post=t5ids(PRE),t5ids(POST); en_ids=t5ids(en); eos=[t5.eos_token_id]
    # target: EN caption both sides
    peE,amE=qwen(PRE+en+POST); outE=run(peE,amE,pre+en_ids+post+eos); T=outE[len(pre):len(pre)+len(en_ids)]; Tm=T.mean(0)
    peJ,amJ=qwen(PRE+ja+POST)                                       # Qwen side reads the JA caption for all arms below
    ceil=run(peJ,amJ,pre+en_ids+post+eos)[len(pre):len(pre)+len(en_ids)].mean(0)
    ja_ext,_=enc.encode(ja,64); ja_ext=[i for i in ja_ext if i>=N0]   # the ext ids for the JA surface
    r_en=base[en_ids].mean(0,keepdim=True); r_ch=table[[i-N0 for i in ja_ext]].float().mean(0,keepdim=True)
    idx=N0+table.shape[0]
    o_en=run(peJ,amJ,pre+[idx]+post+eos,r_en)[len(pre)]; o_ch=run(peJ,amJ,pre+[idx]+post+eos,r_ch)[len(pre)]
    o_pc=run(peJ,amJ,pre+ja_ext+post+eos)[len(pre):len(pre)+len(ja_ext)].mean(0)
    print(f"{ja:6s} {en:14s} |  {float(cos(ceil,Tm)):.2f}                          |  {float(cos(o_en,Tm)):.2f}            |  {float(cos(o_ch,Tm)):.2f}                                |  {float(cos(o_pc,Tm)):.2f}")
