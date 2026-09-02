import torch, torch.nn.functional as F, json, random
from pathlib import Path
from library.anima.weights import load_qwen3_text_encoder, load_t5_tokenizer, load_llm_adapter
from library.anima.strategy import AnimaTokenizeStrategy, AnimaTextEncodingStrategy
from library.anima import ext_vocab
QW='models/text_encoders/qwen_3_06b_base.safetensors'; DIT='models/diffusion_models/anima-base-v1.0.safetensors'
te,qtok=load_qwen3_text_encoder(QW,dtype=torch.float32,device='cpu'); t5=load_t5_tokenizer(None)
ad=load_llm_adapter(DIT,dtype=torch.float32,device='cpu'); tk=AnimaTokenizeStrategy(qtok,t5); es=AnimaTextEncodingStrategy()
base=ad.embed.weight.data.clone(); N0=base.shape[0]
@torch.no_grad()
def run(text,enc):
    ids,mask=enc.encode(text,512); ti=torch.tensor([ids]); tm=torch.tensor([mask])
    q,qm,_,_=tk.tokenize([text]); pe,am,_,_=es.encode_tokens(tk,[te],[q,qm,ti,tm])
    n=int(tm.sum()); out=ad(source_hidden_states=pe,target_input_ids=ti,target_attention_mask=tm,source_attention_mask=am)[0][:n]
    return out, [i>=N0 for i in ids[:n]]
# EN code bank: codes of common tags in a neutral caption
with torch.no_grad():
    stock=ext_vocab.HybridT5Encoder.from_mapping(t5,qtok,{'qwen':{},'char':{}})
    ad.embed=torch.nn.Embedding.from_pretrained(base)
    bank,_=run("1girl, solo, school uniform, cat ears, blonde hair, classroom, smile, maid, cafe, speech bubble, japanese text, comic, monochrome, greyscale, text, sign, sound effects, long hair, blush, bedroom",stock)
    bankn=F.normalize(bank,dim=-1)
random.seed(0); lines=[json.loads(l)['text'] for l in open('post_image_dataset/cjk_unmask/ocr_records_sincos_ppocr.jsonl')]; lines=random.sample([l for l in lines if len(l)>=6],20)
def stats(name):
    if name=='ridge_init':   # untrained ridge-mapped rows: rebuild from build-time assets if present, else skip
        p=Path('bench/cjk_adapter/assets/ext_embed')
        if not p.with_suffix('.safetensors').exists(): return None
    else: p=Path(f'output/ckpt/cjk_vocab_pack_{name}')
    table,mapping=ext_vocab.load_ext_assets(p); enc=ext_vocab.HybridT5Encoder.from_mapping(t5,qtok,mapping)
    ad.embed=torch.nn.Embedding.from_pretrained(torch.cat([base,table.float()]))
    disp=[];inspan=[];normr=[];cross=[]
    with torch.no_grad():
        for l in lines:
            cap=f"1girl, solo, @sincos, japanese text, 「{l}」"
            out,isext=run(cap,enc); E=out[torch.tensor(isext)]; N=out[~torch.tensor(isext)]
            if len(E)<2: continue
            En=F.normalize(E,dim=-1); G=En@En.T; disp.append(float((G.sum()-len(E))/(len(E)**2-len(E))))
            inspan.append(float((En@bankn.T).amax(1).mean()))
            normr.append(float(E.norm(dim=-1).mean()/N.norm(dim=-1).mean()))
            cross.append(float((En@F.normalize(N,dim=-1).T).amax(1).mean()))
    m=lambda x: sum(x)/len(x)
    print(f"{name:12s} ext-ext mean cos {m(disp):.3f} | max cos to EN tag bank {m(inspan):.3f} | max cos to EN slots in same caption {m(cross):.3f} | norm ext/EN {m(normr):.2f}")
print("(reference: EN caption slots ext-ext-equivalent dispersion ≈ 0.05; EN slot vs EN bank max ≈ 1.0 for shared tags)")
for n in ('ridge_init','synthja_v4','synthjako2','synthja_v5'): stats(n)
