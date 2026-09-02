import torch, torch.nn.functional as F, json, collections, glob
from pathlib import Path
from library.anima.weights import load_qwen3_text_encoder, load_t5_tokenizer, load_llm_adapter
from library.anima.strategy import AnimaTokenizeStrategy, AnimaTextEncodingStrategy
from library.anima import ext_vocab
QW='models/text_encoders/qwen_3_06b_base.safetensors'; DIT='models/diffusion_models/anima-base-v1.0.safetensors'
t5=load_t5_tokenizer(None); te,qtok=load_qwen3_text_encoder(QW,dtype=torch.float32,device='cpu'); ad=load_llm_adapter(DIT,dtype=torch.float32,device='cpu')
tk=AnimaTokenizeStrategy(qtok,t5); es=AnimaTextEncodingStrategy(); base=ad.embed.weight.data.clone(); N0=base.shape[0]
table,mapping=ext_vocab.load_ext_assets(Path('output/ckpt/cjk_vocab_pack_synthja_v5')); enc=ext_vocab.HybridT5Encoder.from_mapping(t5,qtok,mapping)
ad.embed=torch.nn.Embedding.from_pretrained(torch.cat([base,table.float()]))
NQ=len(mapping['qwen']); inv={v:k for k,v in mapping['char'].items()}; invq={v:qtok.decode([int(k)]) for k,v in mapping['qwen'].items()}
# JA tag surfaces: from the synthetic pairs file (ja side)
pairs=[]
for p in glob.glob('post_image_dataset/cjk_distill/pairs_synth_tags.jsonl'):
    with open(p) as fh:
        for i,l in enumerate(fh):
            if i>=20000: break
            j=json.loads(l); s=j.get('ja') or j.get('src') or j.get('text_ja') or j.get('cjk') or ''
            if s: pairs.append(s)
print('sample JA captions:',len(pairs), pairs[:2])
cnt=collections.Counter(); kind=collections.Counter()
for s in pairs[:5000]:
    ids,_=enc.encode(s,512)
    for i in ids:
        if i>=N0:
            r=i-N0; k='char' if r>=NQ else 'qwen'; kind[k]+=1
            if k=='char': cnt[inv[r]]+=1
print('ext tokens in tag captions by layer:', dict(kind), f"char share {kind['char']/max(1,sum(kind.values()))*100:.1f}%")
top=[c for c,_ in cnt.most_common(24)]; print('most frequent char-layer kanji:', [(c,n) for c,n in cnt.most_common(24)])
# code-space pairwise cos among those char rows vs among 24 frequent qwen-token kanji rows
q,qm,_,_=tk.tokenize([""]); pe0,am0,_,_=es.encode_tokens(tk,[te],[q,qm,torch.zeros(1,512,dtype=torch.long),torch.zeros(1,512,dtype=torch.long)])
@torch.no_grad()
def codes(ids):
    L=len(ids)+1; ti=torch.full((1,L),t5.pad_token_id); tm=torch.ones(1,L,dtype=torch.long); ti[0,:len(ids)]=torch.tensor(ids); ti[0,-1]=t5.eos_token_id
    return ad(source_hidden_states=pe0,target_input_ids=ti,target_attention_mask=tm,source_attention_mask=am0)[0][:len(ids)]
def stats(M):
    Mn=F.normalize(M,dim=-1); G=Mn@Mn.T; n=len(M); off=G[~torch.eye(n,dtype=bool)]; return f"pairwise cos {off.mean():.3f}, min {off.min():.2f}, >0.5 {float((off>0.5).float().mean()*100):.0f}%"
char_ids=[N0+mapping['char'][c] for c in top]
qcnt=collections.Counter()
for s in pairs[:5000]:
    for i in enc.encode(s,512)[0]:
        if N0<=i<N0+NQ and len(invq[i-N0].strip())==1 and '一'<=invq[i-N0].strip()<='鿿': qcnt[i]+=1
q_ids=[i for i,_ in qcnt.most_common(24)]
print('keys  char-layer kanji:', stats(table[[i-N0 for i in char_ids]].float()), '| qwen-token kanji:', stats(table[[i-N0 for i in q_ids]].float()))
print('codes char-layer kanji:', stats(codes(char_ids)), '| qwen-token kanji:', stats(codes(q_ids)))
print('qwen-token kanji used as control:', [invq[i-N0] for i in q_ids])
