from __future__ import annotations
_S='passthrough_fp16'
_R='passthrough_ctrl'
_Q='passthrough'
_P='.proj.'
_O='momentum'
_N='INT6_STE'
_M='fineweb_train_*.bin'
_L='.scale'
_K='zstd'
_J='mlp'
_I='utf-8'
_H='params'
_G='cuda'
_F='lr'
_E=.0
_D=1.
_C=False
_B=True
_A=None
import copy,glob,io,math,os,random,subprocess,sys,time,uuid,zlib
from pathlib import Path
try:import zstandard;_COMPRESSOR=_K
except ImportError:_COMPRESSOR='zlib'
import numpy as np,sentencepiece as spm,torch,torch.distributed as dist,torch.nn.functional as F
from torch import Tensor,nn
from torch.nn.parallel import DistributedDataParallel as DDP
class Hyperparameters:data_path=os.environ.get('DATA_PATH','./data/datasets/fineweb10B_sp1024');train_files=os.path.join(data_path,_M);val_files=os.path.join(data_path,'fineweb_val_*.bin');tokenizer_path=os.environ.get('TOKENIZER_PATH','./data/tokenizers/fineweb_1024_bpe.model');run_id=os.environ.get('RUN_ID',str(uuid.uuid4()));seed=int(os.environ.get('SEED',1337));val_batch_size=int(os.environ.get('VAL_BATCH_SIZE',524288));val_loss_every=int(os.environ.get('VAL_LOSS_EVERY',500));train_log_every=int(os.environ.get('TRAIN_LOG_EVERY',100));iterations=int(os.environ.get('ITERATIONS',20000));warmdown_iters=int(os.environ.get('WARMDOWN_ITERS',3000));warmup_steps=int(os.environ.get('WARMUP_STEPS',20));train_batch_tokens=int(os.environ.get('TRAIN_BATCH_TOKENS',786432));train_seq_len=int(os.environ.get('TRAIN_SEQ_LEN',2048));max_wallclock_seconds=float(os.environ.get('MAX_WALLCLOCK_SECONDS',6e2));qk_gain_init=float(os.environ.get('QK_GAIN_INIT',1.5));vocab_size=int(os.environ.get('VOCAB_SIZE',1024));num_layers=int(os.environ.get('NUM_LAYERS',9));num_kv_heads=int(os.environ.get('NUM_KV_HEADS',4));model_dim=int(os.environ.get('MODEL_DIM',512));num_heads=int(os.environ.get('NUM_HEADS',8));mlp_mult=float(os.environ.get('MLP_MULT',3.));tie_embeddings=bool(int(os.environ.get('TIE_EMBEDDINGS','1')));rope_base=float(os.environ.get('ROPE_BASE',1e4));logit_softcap=float(os.environ.get('LOGIT_SOFTCAP',3e1));embed_lr=float(os.environ.get('EMBED_LR',.6));head_lr=float(os.environ.get('HEAD_LR',.008));tied_embed_lr=float(os.environ.get('TIED_EMBED_LR',.03));tied_embed_init_std=float(os.environ.get('TIED_EMBED_INIT_STD',.005));matrix_lr=float(os.environ.get('MATRIX_LR',.02));scalar_lr=float(os.environ.get('SCALAR_LR',.02));muon_momentum=float(os.environ.get('MUON_MOMENTUM',.99));muon_backend_steps=int(os.environ.get('MUON_BACKEND_STEPS',5));muon_momentum_warmup_start=float(os.environ.get('MUON_MOMENTUM_WARMUP_START',.92));muon_momentum_warmup_steps=int(os.environ.get('MUON_MOMENTUM_WARMUP_STEPS',1500));beta1=float(os.environ.get('BETA1',.9));beta2=float(os.environ.get('BETA2',.95));adam_eps=float(os.environ.get('ADAM_EPS',1e-08));grad_clip_norm=float(os.environ.get('GRAD_CLIP_NORM',.3));weight_decay=float(os.environ.get('WEIGHT_DECAY',.01));muon_wd=float(os.environ.get('MUON_WD',.02));eval_stride=int(os.environ.get('EVAL_STRIDE',64));eval_batch_seqs=int(os.environ.get('EVAL_BATCH_SEQS',32));bigram_vocab_size=int(os.environ.get('BIGRAM_VOCAB_SIZE',2048));bigram_dim=int(os.environ.get('BIGRAM_DIM',64));int6_ste=bool(int(os.environ.get(_N,'0')));swa_enabled=bool(int(os.environ.get('SWA_ENABLED','1')));swa_start_frac=float(os.environ.get('SWA_START_FRAC',.5));swa_every=int(os.environ.get('SWA_EVERY',50));prune_frac=float(os.environ.get('PRUNE_FRAC',.02));int5_mlp=bool(int(os.environ.get('INT5_MLP','1')));ttt_enabled=bool(int(os.environ.get('TTT_ENABLED','1')));ttt_lr=float(os.environ.get('TTT_LR',.004));ttt_epochs=int(os.environ.get('TTT_EPOCHS',2));ttt_momentum=float(os.environ.get('TTT_MOMENTUM',.9));ttt_batch_seqs=int(os.environ.get('TTT_BATCH_SEQS',32));ttt_freeze_layers=int(os.environ.get('TTT_FREEZE_LAYERS',4))
def zeropower_via_newtonschulz5(G,steps=10,eps=1e-07):
	D,E,F=3.4445,-4.775,2.0315;A=G.bfloat16();A/=A.norm()+eps;C=G.size(0)>G.size(1)
	if C:A=A.T
	for I in range(steps):B=A@A.T;H=E*B+F*B@B;A=D*A+H@A
	return A.T if C else A
class Muon(torch.optim.Optimizer):
	def __init__(A,params,lr,momentum,backend_steps,nesterov=_B,weight_decay=_E):super().__init__(params,dict(lr=lr,momentum=momentum,backend_steps=backend_steps,nesterov=nesterov,weight_decay=weight_decay))
	@torch.no_grad()
	def step(self,closure=_A):
		J=closure;I='momentum_buffer';K=_A
		if J is not _A:
			with torch.enable_grad():K=J()
		F=dist.is_available()and dist.is_initialized();P=dist.get_world_size()if F else 1;Q=dist.get_rank()if F else 0
		for D in self.param_groups:
			E=D[_H]
			if not E:continue
			L=D[_F];M=D[_O];R=D['backend_steps'];S=D['nesterov'];T=sum(int(A.numel())for A in E);G=torch.zeros(T,device=E[0].device,dtype=torch.bfloat16);C=0
			for(U,A)in enumerate(E):
				if U%P==Q and A.grad is not _A:
					B=A.grad;H=self.state[A]
					if I not in H:H[I]=torch.zeros_like(B)
					N=H[I];N.mul_(M).add_(B)
					if S:B=B.add(N,alpha=M)
					B=zeropower_via_newtonschulz5(B,steps=R);B*=max(1,B.size(0)/B.size(1))**.5;G[C:C+A.numel()]=B.reshape(-1)
				C+=A.numel()
			if F:dist.all_reduce(G,op=dist.ReduceOp.SUM)
			O=D.get('weight_decay',_E);C=0
			for A in E:
				B=G[C:C+A.numel()].view_as(A).to(dtype=A.dtype)
				if O>0:A.data.mul_(_D-L*O)
				A.add_(B,alpha=-L);C+=A.numel()
		return K
def build_sentencepiece_luts(sp,vocab_size,device):
	D=device;B=sp;G=int(B.vocab_size());E=max(G,vocab_size);F=np.zeros((E,),dtype=np.int16);H=np.zeros((E,),dtype=np.bool_);I=np.ones((E,),dtype=np.bool_)
	for A in range(G):
		if B.is_control(A)or B.is_unknown(A)or B.is_unused(A):continue
		I[A]=_C
		if B.is_byte(A):F[A]=1;continue
		C=B.id_to_piece(A)
		if C.startswith('▁'):H[A]=_B;C=C[1:]
		F[A]=len(C.encode(_I))
	return torch.tensor(F,dtype=torch.int16,device=D),torch.tensor(H,dtype=torch.bool,device=D),torch.tensor(I,dtype=torch.bool,device=D)
def load_validation_tokens(pattern,seq_len):
	B=pattern;A=seq_len;C=[Path(A)for A in sorted(glob.glob(B))]
	if not C:raise FileNotFoundError(f"No files found for pattern: {B}")
	D=torch.cat([load_data_shard(A)for A in C]).contiguous();E=(D.numel()-1)//A*A
	if E<=0:raise ValueError(f"Validation split is too short for TRAIN_SEQ_LEN={A}")
	return D[:E+1]
def eval_val(args,model,rank,world_size,device,grad_accum_steps,val_tokens,base_bytes_lut,has_leading_space_lut,is_boundary_token_lut):
	J=val_tokens;I=grad_accum_steps;E=model;C=device;B=world_size;A=args;K=A.val_batch_size//(B*I)
	if K<A.train_seq_len:raise ValueError(f"VAL_BATCH_SIZE must provide at least one sequence per rank; got VAL_BATCH_SIZE={A.val_batch_size}, WORLD_SIZE={B}, GRAD_ACCUM_STEPS={I}, TRAIN_SEQ_LEN={A.train_seq_len}")
	L=K//A.train_seq_len;M=(J.numel()-1)//A.train_seq_len;V=M*rank//B;N=M*(rank+1)//B;F=torch.zeros((),device=C,dtype=torch.float64);D=torch.zeros((),device=C,dtype=torch.float64);G=torch.zeros((),device=C,dtype=torch.float64);E.eval()
	with torch.inference_mode():
		for O in range(V,N,L):
			W=min(O+L,N);X=O*A.train_seq_len;Y=W*A.train_seq_len+1;P=J[X:Y].to(device=C,dtype=torch.int64,non_blocking=_B);Q=P[:-1].reshape(-1,A.train_seq_len);H=P[1:].reshape(-1,A.train_seq_len)
			with torch.autocast(device_type=_G,dtype=torch.bfloat16,enabled=_B):Z=E(Q,H).detach()
			R=float(H.numel());F+=Z.to(torch.float64)*R;D+=R;a=Q.reshape(-1);S=H.reshape(-1);T=base_bytes_lut[S].to(dtype=torch.int16);T+=(has_leading_space_lut[S]&~is_boundary_token_lut[a]).to(dtype=torch.int16);G+=T.to(torch.float64).sum()
	if dist.is_available()and dist.is_initialized():dist.all_reduce(F,op=dist.ReduceOp.SUM);dist.all_reduce(D,op=dist.ReduceOp.SUM);dist.all_reduce(G,op=dist.ReduceOp.SUM)
	U=F/D;b=U.item()/math.log(2.);c=D.item()/G.item();E.train();return float(U.item()),float(b*c)
CONTROL_TENSOR_NAME_PATTERNS=tuple(A for A in os.environ.get('CONTROL_TENSOR_NAME_PATTERNS','attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights,smear,bigram.scale').split(',')if A)
FP16_KEEP_NAME_PATTERNS=tuple(A for A in os.environ.get('FP16_KEEP_NAME_PATTERNS','tok_emb,blocks.8.attn.c_k').split(',')if A)
INT8_KEEP_FLOAT_FP32_NAME_PATTERNS=tuple(A for A in os.environ.get('INT8_KEEP_FLOAT_FP32_NAME_PATTERNS',','.join(CONTROL_TENSOR_NAME_PATTERNS)).split(',')if A)
INT8_KEEP_FLOAT_MAX_NUMEL=65536
INT8_KEEP_FLOAT_STORE_DTYPE=torch.float16
INT8_PER_ROW_SCALE_DTYPE=torch.float16
INT8_CLIP_PERCENTILE=99.99984
INT8_CLIP_Q=INT8_CLIP_PERCENTILE/1e2
def tensor_nbytes(t):return int(t.numel())*int(t.element_size())
def quantize_float_tensor(t):
	A=t.float()
	if A.ndim==2:B=torch.quantile(A.abs(),INT8_CLIP_Q,dim=1)if A.numel()else torch.empty((A.shape[0],),dtype=torch.float32);E=torch.maximum(torch.minimum(A,B[:,_A]),-B[:,_A]);C=(B/127.).clamp_min(_D/127.);D=torch.clamp(torch.round(E/C[:,_A]),-127,127).to(torch.int8).contiguous();return D,C.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()
	B=float(torch.quantile(A.abs().flatten(),INT8_CLIP_Q).item())if A.numel()else _E;C=torch.tensor(B/127. if B>0 else _D,dtype=torch.float32);D=torch.clamp(torch.round(torch.clamp(A,-B,B)/C),-127,127).to(torch.int8).contiguous();return D,C
def _classify_param(name):
	B='.mlp.';A=name
	if'tok_emb'in A or'lm_head'in A:return'embed'
	if B in A:return _J
	if'.attn.'in A or _P in A and B not in A:return'attn'
	return'other'
def quantize_intN_per_row(t,clip_range=31):
	B=clip_range;C=t.float()
	if C.ndim==2:E=C.abs().amax(dim=1);A=(E/B).clamp_min(1e-12).to(torch.float16);A=A.clamp_min(torch.finfo(torch.float16).tiny);D=torch.clamp(torch.round(C/A.float()[:,_A]),-(B+1),B).to(torch.int8);return D,A
	F=C.abs().max().item();A=torch.tensor(max(F/B,1e-12),dtype=torch.float16);D=torch.clamp(torch.round(C/A.float()),-(B+1),B).to(torch.int8);return D,A
def mixed_quantize_int6(state_dict,int6_cats):
	H='type';C={};D={}
	for(A,I)in state_dict.items():
		B=I.detach().cpu().contiguous();E=_classify_param(A)
		if not B.is_floating_point()or B.numel()<=65536:C[A]=B.to(torch.float16)if B.is_floating_point()else B;D[A]=_Q;continue
		if any(B in A for B in CONTROL_TENSOR_NAME_PATTERNS):C[A]=B.float();D[A]=_R;continue
		if any(B in A for B in FP16_KEEP_NAME_PATTERNS):C[A]=B.to(dtype=torch.float16).contiguous();D[A]=_S;continue
		if E in int6_cats and B.ndim>=1:J=15 if E==_J else 31;F,G=quantize_intN_per_row(B,clip_range=J);C[A+'.q']=F;C[A+_L]=G;D[A]={H:f"int{5 if E==_J else 6}"}
		else:F,G=quantize_float_tensor(B);C[A+'.q']=F;C[A+_L]=G;D[A]={H:'int8'}
	return C,D
def dequantize_mixed_int6(result,meta,template_sd):
	F=result;B={}
	for(A,H)in template_sd.items():
		I=meta[A];C=H.dtype
		if I in(_Q,_R,_S):
			D=F[A]
			if D.dtype==torch.float16 and C in(torch.float32,torch.bfloat16):D=D.to(C)
			B[A]=D;continue
		E,G=F[A+'.q'],F[A+_L]
		if G.ndim>0:B[A]=(E.float()*G.float().view(E.shape[0],*[1]*(E.ndim-1))).to(C)
		else:B[A]=(E.float()*float(G.item())).to(C)
	return B
def load_data_shard(file):
	H='<u2';G='<i4';A=file;D=256*np.dtype(G).itemsize;I=np.dtype(H).itemsize;B=np.fromfile(A,dtype=G,count=256)
	if B.size!=256 or int(B[0])!=20240520 or int(B[1])!=1:raise ValueError(f"Unexpected shard header for {A}")
	C=int(B[2]);E=D+C*I
	if A.stat().st_size!=E:raise ValueError(f"Shard size mismatch for {A}: expected {E} bytes")
	F=np.fromfile(A,dtype=H,count=C,offset=D)
	if F.size!=C:raise ValueError(f"Short read for {A}")
	return torch.from_numpy(F.astype(np.uint16,copy=_C))
class TokenStream:
	def __init__(A,pattern):
		B=pattern;A.files=[Path(A)for A in sorted(glob.glob(B))]
		if not A.files:raise FileNotFoundError(f"No files found for pattern: {B}")
		A.file_idx=0;A.tokens=load_data_shard(A.files[0]);A.pos=0
	def _advance_file(A):A.file_idx=(A.file_idx+1)%len(A.files);A.tokens=load_data_shard(A.files[A.file_idx]);A.pos=0
	def take(A,n):
		B=[];C=n
		while C>0:
			E=A.tokens.numel()-A.pos
			if E<=0:A._advance_file();continue
			D=min(C,E);B.append(A.tokens[A.pos:A.pos+D]);A.pos+=D;C-=D
		return B[0]if len(B)==1 else torch.cat(B)
class DistributedTokenLoader:
	def __init__(A,pattern,rank,world_size,device):A.rank=rank;A.world_size=world_size;A.device=device;A.stream=TokenStream(pattern)
	def next_batch(A,global_tokens,seq_len,grad_accum_steps):C=seq_len;F=global_tokens//(A.world_size*grad_accum_steps);B=F+1;G=A.stream.take(B*A.world_size);D=A.rank*B;E=G[D:D+B].to(dtype=torch.int64);H=E[:-1].reshape(-1,C);I=E[1:].reshape(-1,C);return H.to(A.device,non_blocking=_B),I.to(A.device,non_blocking=_B)
class RMSNorm(nn.Module):
	def __init__(A,eps=_A):super().__init__();A.eps=eps
	def forward(A,x):return F.rms_norm(x,(x.size(-1),),eps=A.eps)
INT6_QUANT_RANGE=31
INT6_CLIP_Q=.9999984
_INT6_STE_ENABLED=bool(int(os.environ.get(_N,'0')))
class CastedLinear(nn.Linear):
	def forward(B,x):
		A=B.weight.to(x.dtype)
		if _INT6_STE_ENABLED and B.training and A.ndim==2:
			with torch.no_grad():D=A.float();C=torch.quantile(D.abs(),INT6_CLIP_Q,dim=1).clamp_min(1e-08);E=C/INT6_QUANT_RANGE;G=torch.clamp(D,-C[:,_A],C[:,_A]);H=(torch.round(G/E[:,_A])*E[:,_A]).to(x.dtype)
			A=A+(H-A).detach()
		I=B.bias.to(x.dtype)if B.bias is not _A else _A;return F.linear(x,A,I)
def restore_low_dim_params_to_fp32(module):
	with torch.no_grad():
		for(B,A)in module.named_parameters():
			if(A.ndim<2 or any(A in B for A in CONTROL_TENSOR_NAME_PATTERNS))and A.dtype!=torch.float32:A.data=A.data.float()
class Rotary(nn.Module):
	def __init__(A,dim,base=1e4):super().__init__();B=_D/base**(torch.arange(0,dim,2,dtype=torch.float32)/dim);A.register_buffer('inv_freq',B,persistent=_C);A._seq_len_cached=0;A._cos_cached=_A;A._sin_cached=_A
	def forward(A,seq_len,device,dtype):
		D=dtype;C=device;B=seq_len
		if A._cos_cached is _A or A._sin_cached is _A or A._seq_len_cached!=B or A._cos_cached.device!=C:F=torch.arange(B,device=C,dtype=A.inv_freq.dtype);E=torch.outer(F,A.inv_freq.to(C));A._cos_cached=E.cos()[_A,_A,:,:];A._sin_cached=E.sin()[_A,_A,:,:];A._seq_len_cached=B
		return A._cos_cached.to(dtype=D),A._sin_cached.to(dtype=D)
def apply_rotary_emb(x,cos,sin):A=x.size(-1)//2;B,C=x[...,:A],x[...,A:];return torch.cat((B*cos+C*sin,B*-sin+C*cos),dim=-1)
class CausalSelfAttention(nn.Module):
	def __init__(A,dim,num_heads,num_kv_heads,rope_base,qk_gain_init):
		D=num_kv_heads;C=num_heads;B=dim;super().__init__()
		if B%C!=0:raise ValueError('model_dim must be divisible by num_heads')
		if C%D!=0:raise ValueError('num_heads must be divisible by num_kv_heads')
		A.num_heads=C;A.num_kv_heads=D;A.head_dim=B//C
		if A.head_dim%2!=0:raise ValueError('head_dim must be even for RoPE')
		E=A.num_kv_heads*A.head_dim;A.c_q=CastedLinear(B,B,bias=_C);A.c_k=CastedLinear(B,E,bias=_C);A.c_v=CastedLinear(B,E,bias=_C);A.proj=CastedLinear(B,B,bias=_C);A.proj._zero_init=_B;A.q_gain=nn.Parameter(torch.full((C,),qk_gain_init,dtype=torch.float32));A.rotary=Rotary(A.head_dim,base=rope_base);F=bool(int(os.environ.get('ATTN_GATE_ENABLED','1')));A.attn_gate=CastedLinear(min(12,B),C,bias=_C)if F else _A
		if A.attn_gate is not _A:nn.init.zeros_(A.attn_gate.weight)
	def forward(A,x):
		G,D,J=x.shape;B=A.c_q(x).reshape(G,D,A.num_heads,A.head_dim).transpose(1,2);C=A.c_k(x).reshape(G,D,A.num_kv_heads,A.head_dim).transpose(1,2);K=A.c_v(x).reshape(G,D,A.num_kv_heads,A.head_dim).transpose(1,2);B=F.rms_norm(B,(B.size(-1),));C=F.rms_norm(C,(C.size(-1),));H,I=A.rotary(D,x.device,B.dtype);B=apply_rotary_emb(B,H,I);C=apply_rotary_emb(C,H,I);B=B*A.q_gain.to(dtype=B.dtype)[_A,:,_A,_A];E=F.scaled_dot_product_attention(B,C,K,attn_mask=_A,is_causal=_B,enable_gqa=A.num_kv_heads!=A.num_heads)
		if A.attn_gate is not _A:L=x[...,:A.attn_gate.weight.size(-1)];M=torch.sigmoid(A.attn_gate(L));E=E*M.unsqueeze(-1).transpose(1,2)
		E=E.transpose(1,2).contiguous().reshape(G,D,J);return A.proj(E)
class MLP(nn.Module):
	def __init__(A,dim,mlp_mult):B=dim;super().__init__();C=int(mlp_mult*B);A.fc=CastedLinear(B,C,bias=_C);A.proj=CastedLinear(C,B,bias=_C);A.proj._zero_init=_B
	def forward(A,x):x=torch.relu(A.fc(x));return A.proj(x.square())
class SmearGate(nn.Module):
	def __init__(A,dim):super().__init__();A.gate=nn.Parameter(torch.zeros(dim,dtype=torch.float32))
	def forward(B,x):A=torch.sigmoid(B.gate.to(dtype=x.dtype))[_A,_A,:];C=torch.cat([torch.zeros_like(x[:,:1]),x[:,:-1]],dim=1);return(1-A)*x+A*C
class BigramHashEmbedding(nn.Module):
	def __init__(A,bigram_vocab_size,bigram_dim,model_dim):
		D=model_dim;C=bigram_vocab_size;B=bigram_dim;super().__init__();A.bigram_vocab_size=C;A.embed=nn.Embedding(C,B);nn.init.zeros_(A.embed.weight);A.proj=CastedLinear(B,D,bias=_C)if B!=D else _A
		if A.proj is not _A:nn.init.zeros_(A.proj.weight)
		A.scale=nn.Parameter(torch.tensor(.05,dtype=torch.float32))
	def bigram_hash(D,tokens):A=tokens.to(torch.int32);C=D.bigram_vocab_size-1;B=torch.empty_like(A);B[...,0]=C;B[...,1:]=torch.bitwise_xor(36313*A[...,1:],27191*A[...,:-1])%C;return B.long()
	def forward(A,token_ids):
		B=A.embed(A.bigram_hash(token_ids))
		if A.proj is not _A:B=A.proj(B)
		return B*A.scale.to(dtype=B.dtype)
class Block(nn.Module):
	def __init__(A,dim,num_heads,num_kv_heads,mlp_mult,rope_base,qk_gain_init):B=dim;super().__init__();A.attn_norm=RMSNorm();A.mlp_norm=RMSNorm();A.attn=CausalSelfAttention(B,num_heads,num_kv_heads,rope_base,qk_gain_init);A.mlp=MLP(B,mlp_mult);A.attn_scale=nn.Parameter(torch.ones(B,dtype=torch.float32));A.mlp_scale=nn.Parameter(torch.ones(B,dtype=torch.float32));A.resid_mix=nn.Parameter(torch.stack((torch.ones(B),torch.zeros(B))).float())
	def forward(A,x,x0):B=A.resid_mix.to(dtype=x.dtype);x=B[0][_A,_A,:]*x+B[1][_A,_A,:]*x0;C=A.attn(A.attn_norm(x));x=x+A.attn_scale.to(dtype=x.dtype)[_A,_A,:]*C;x=x+A.mlp_scale.to(dtype=x.dtype)[_A,_A,:]*A.mlp(A.mlp_norm(x));return x
class GPT(nn.Module):
	def __init__(A,vocab_size,num_layers,model_dim,num_heads,num_kv_heads,mlp_mult,tie_embeddings,tied_embed_init_std,logit_softcap,rope_base,qk_gain_init,bigram_vocab_size=0,bigram_dim=128):
		G=bigram_vocab_size;F=tie_embeddings;E=vocab_size;D=logit_softcap;C=num_layers;B=model_dim;super().__init__()
		if D<=_E:raise ValueError(f"logit_softcap must be positive, got {D}")
		A.tie_embeddings=F;A.tied_embed_init_std=tied_embed_init_std;A.logit_softcap=D;A.tok_emb=nn.Embedding(E,B);A.bigram=BigramHashEmbedding(G,bigram_dim,B)if G>0 else _A;A.num_encoder_layers=C//2;A.num_decoder_layers=C-A.num_encoder_layers;A.num_skip_weights=min(A.num_encoder_layers,A.num_decoder_layers);A.skip_weights=nn.Parameter(torch.ones(A.num_skip_weights,B,dtype=torch.float32));A.smear=SmearGate(B);A.blocks=nn.ModuleList([Block(B,num_heads,num_kv_heads,mlp_mult,rope_base,qk_gain_init)for A in range(C)]);A.final_norm=RMSNorm();A.lm_head=_A if F else CastedLinear(B,E,bias=_C)
		if A.lm_head is not _A:A.lm_head._zero_init=_B
		A._init_weights()
	def _init_weights(B):
		if B.tie_embeddings:nn.init.normal_(B.tok_emb.weight,mean=_E,std=B.tied_embed_init_std)
		D=len(B.blocks)
		for(C,A)in B.named_modules():
			if isinstance(A,nn.Linear):
				if getattr(A,'_zero_init',_C):nn.init.zeros_(A.weight)
				elif A.weight.ndim==2 and A.weight.shape[0]>=64 and A.weight.shape[1]>=64:
					nn.init.orthogonal_(A.weight,gain=_D)
					if _P in C or C.endswith('.proj'):
						with torch.no_grad():A.weight.mul_(_D/math.sqrt(2*D))
	def forward(B,input_ids,target_ids):
		E=input_ids;A=B.tok_emb(E)
		if B.bigram is not _A:A=A+B.bigram(E)
		A=F.rms_norm(A,(A.size(-1),));A=B.smear(A);G=A;D=[]
		for C in range(B.num_encoder_layers):A=B.blocks[C](A,G);D.append(A)
		for C in range(B.num_decoder_layers):
			if D:A=A+B.skip_weights[C].to(dtype=A.dtype)[_A,_A,:]*D.pop()
			A=B.blocks[B.num_encoder_layers+C](A,G)
		A=B.final_norm(A).reshape(-1,A.size(-1));I=target_ids.reshape(-1)
		if B.tie_embeddings:H=F.linear(A,B.tok_emb.weight)
		else:
			if B.lm_head is _A:raise RuntimeError('lm_head is required when tie_embeddings=False')
			H=B.lm_head(A)
		J=B.logit_softcap*torch.tanh(H/B.logit_softcap);return F.cross_entropy(J.float(),I,reduction='mean')
	def forward_logits(B,input_ids):
		E=input_ids;A=B.tok_emb(E)
		if B.bigram is not _A:A=A+B.bigram(E)
		A=F.rms_norm(A,(A.size(-1),));A=B.smear(A);G=A;D=[]
		for C in range(B.num_encoder_layers):A=B.blocks[C](A,G);D.append(A)
		for C in range(B.num_decoder_layers):
			if D:A=A+B.skip_weights[C].to(dtype=A.dtype)[_A,_A,:]*D.pop()
			A=B.blocks[B.num_encoder_layers+C](A,G)
		A=B.final_norm(A)
		if B.tie_embeddings:H=F.linear(A,B.tok_emb.weight)
		else:H=B.lm_head(A)
		return B.logit_softcap*torch.tanh(H/B.logit_softcap)
def ttt_adapt(args,base_model,device,val_tokens,rank=0,world_size=1,log_fn=_A):
	M=val_tokens;G=device;E=log_fn;D=world_size;C=base_model;A=args;B=A.train_seq_len;N=(M.numel()-1)//B;O=A.ttt_batch_seqs;P=A.ttt_freeze_layers
	if P>0:
		for(W,X)in enumerate(C.blocks):
			if W<P:
				for F in X.parameters():F.requires_grad_(_C)
	H=[A for A in C.parameters()if A.requires_grad];Q=torch.optim.SGD(H,lr=A.ttt_lr,momentum=A.ttt_momentum);Y=N*rank//D;R=N*(rank+1)//D;C.train();S=time.perf_counter()
	for Z in range(A.ttt_epochs):
		I=torch.zeros((),device=G,dtype=torch.float64);J=torch.zeros((),device=G,dtype=torch.float64)
		for T in range(Y,R,O):
			a=min(T+O,R);b=T*B;c=a*B+1;U=M[b:c].to(device=G,dtype=torch.int64,non_blocking=_B);d=U[:-1].reshape(-1,B);K=U[1:].reshape(-1,B);Q.zero_grad(set_to_none=_B)
			with torch.autocast(device_type=_G,dtype=torch.bfloat16,enabled=_B):V=C(d,K)
			V.backward()
			if D>1:
				for F in H:
					if F.grad is not _A:dist.all_reduce(F.grad,op=dist.ReduceOp.AVG)
			torch.nn.utils.clip_grad_norm_(H,_D);Q.step();I+=V.detach().to(torch.float64)*K.numel();J+=float(K.numel())
		if D>1:dist.all_reduce(I,op=dist.ReduceOp.SUM);dist.all_reduce(J,op=dist.ReduceOp.SUM)
		L=time.perf_counter()-S;e=I.item()/max(J.item(),1)
		if E:E(f"ttt_epoch:{Z+1}/{A.ttt_epochs} loss:{e:.4f} time:{L:.1f}s")
	L=time.perf_counter()-S
	if E:E(f"ttt:done elapsed={L:.1f}s")
def eval_val_sliding(args,base_model,rank,world_size,device,val_tokens,base_bytes_lut,has_leading_space_lut,is_boundary_token_lut,stride,batch_seqs=32):
	W=stride;V=val_tokens;U=world_size;O=rank;N=base_model;I=batch_seqs;D=device;E=args.train_seq_len;P=V.numel()-1;X=[A for A in range(0,P,W)if min(A+E,P)-A>=1];Y=len(X);i=Y*O//U;j=Y*(O+1)//U;G=X[i:j];J=torch.zeros((),device=D,dtype=torch.float64);B=torch.zeros((),device=D,dtype=torch.float64);K=torch.zeros((),device=D,dtype=torch.float64);N.eval()
	with torch.inference_mode():
		for L in range(0,len(G),I):
			Q=G[L:L+I];R=len(Q);S=torch.zeros(R,E,dtype=torch.int64,device=D);T=torch.zeros(R,E,dtype=torch.int64,device=D);Z=[]
			for(C,H)in enumerate(Q):a=min(H+E,P);A=a-H;Z.append(A);b=V[H:a+1].to(dtype=torch.int64,device=D);S[C,:A]=b[:-1];T[C,:A]=b[1:]
			with torch.autocast(device_type=_G,dtype=torch.bfloat16):c=N.forward_logits(S)
			k=F.cross_entropy(c.reshape(-1,c.size(-1)).float(),T.reshape(-1),reduction='none').reshape(R,E)
			for(C,H)in enumerate(Q):A=Z[C];M=0 if H==0 else max(A-W,0);l=k[C,M:A].to(torch.float64);J+=l.sum();B+=float(A-M);d=T[C,M:A];m=S[C,M:A];e=base_bytes_lut[d].to(torch.float64);e+=(has_leading_space_lut[d]&~is_boundary_token_lut[m]).to(torch.float64);K+=e.sum()
			if O==0 and L//I%50==0:
				f=min(L+I,len(G));n=f/len(G)*100;g=_E
				if B.item()>0:o=(J/B).item();g=o/math.log(2.)*(B.item()/K.item())
				print(f"  sliding_eval [{n:5.1f}%] {f}/{len(G)} windows running_bpb={g:.6f}",flush=_B)
	if dist.is_available()and dist.is_initialized():dist.all_reduce(J,op=dist.ReduceOp.SUM);dist.all_reduce(B,op=dist.ReduceOp.SUM);dist.all_reduce(K,op=dist.ReduceOp.SUM)
	h=(J/B).item();p=h/math.log(2.);q=B.item()/K.item();N.train();return h,p*q
def main():
	AM='final_model.pt';AL='WORLD_SIZE';v='final_model.int8.ptz';R='base_lr';global zeropower_via_newtonschulz5;c=Path(__file__).read_text(encoding=_I);A=Hyperparameters();zeropower_via_newtonschulz5=torch.compile(zeropower_via_newtonschulz5);H='RANK'in os.environ and AL in os.environ;J=int(os.environ.get('RANK','0'));E=int(os.environ.get(AL,'1'));w=int(os.environ.get('LOCAL_RANK','0'))
	if E<=0:raise ValueError(f"WORLD_SIZE must be positive, got {E}")
	if 8%E!=0:raise ValueError(f"WORLD_SIZE={E} must divide 8 so grad_accum_steps stays integral")
	G=8//E;x=_D/G
	if not torch.cuda.is_available():raise RuntimeError('CUDA is required')
	F=torch.device(_G,w);torch.cuda.set_device(F)
	if H:dist.init_process_group(backend='nccl',device_id=F);dist.barrier()
	K=J==0;torch.backends.cuda.matmul.allow_tf32=_B;torch.backends.cudnn.allow_tf32=_B;from torch.backends.cuda import enable_cudnn_sdp as AN,enable_flash_sdp as AO,enable_math_sdp as AP,enable_mem_efficient_sdp as AQ;AN(_C);AO(_B);AQ(_C);AP(_C);Y=_A
	if K:os.makedirs('logs',exist_ok=_B);Y=f"logs/{A.run_id}.txt";print(Y)
	def B(msg,console=_B):
		if not K:return
		if console:print(msg)
		if Y is not _A:
			with open(Y,'a',encoding=_I)as A:print(msg,file=A)
	B(c,console=_C);B('='*100,console=_C);B(f"Running Python {sys.version}",console=_C);B(f"Running PyTorch {torch.__version__}",console=_C);B(subprocess.run(['nvidia-smi'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=_B,check=_C).stdout,console=_C);B('='*100,console=_C);random.seed(A.seed);np.random.seed(A.seed);torch.manual_seed(A.seed);torch.cuda.manual_seed_all(A.seed)
	if not A.tokenizer_path.endswith('.model'):raise ValueError(f"Script only setup for SentencePiece .model file: {A.tokenizer_path}")
	d=spm.SentencePieceProcessor(model_file=A.tokenizer_path)
	if int(d.vocab_size())!=A.vocab_size:raise ValueError(f"VOCAB_SIZE={A.vocab_size} does not match tokenizer vocab_size={int(d.vocab_size())}")
	y=Path(A.data_path).resolve();AR=len(list(y.glob(_M)));S=load_validation_tokens(A.val_files,A.train_seq_len);e,f,g=build_sentencepiece_luts(d,A.vocab_size,F);B(f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={A.tokenizer_path}");B(f"train_loader:dataset:{y.name} train_shards:{AR}");B(f"val_loader:shards pattern={A.val_files} tokens:{S.numel()-1}");C=GPT(vocab_size=A.vocab_size,num_layers=A.num_layers,model_dim=A.model_dim,num_heads=A.num_heads,num_kv_heads=A.num_kv_heads,mlp_mult=A.mlp_mult,tie_embeddings=A.tie_embeddings,tied_embed_init_std=A.tied_embed_init_std,logit_softcap=A.logit_softcap,rope_base=A.rope_base,qk_gain_init=A.qk_gain_init,bigram_vocab_size=A.bigram_vocab_size,bigram_dim=A.bigram_dim).to(F).bfloat16()
	for z in C.modules():
		if isinstance(z,CastedLinear):z.float()
	restore_low_dim_params_to_fp32(C);A0=torch.compile(C,dynamic=_C,fullgraph=_B);I=DDP(A0,device_ids=[w],broadcast_buffers=_C)if H else A0;A1=list(C.blocks.named_parameters());A2=[A for(B,A)in A1 if A.ndim==2 and not any(A in B for A in CONTROL_TENSOR_NAME_PATTERNS)];Z=[A for(B,A)in A1 if A.ndim<2 or any(A in B for A in CONTROL_TENSOR_NAME_PATTERNS)]
	if C.skip_weights.numel()>0:Z.append(C.skip_weights)
	Z.append(C.smear.gate)
	if C.bigram is not _A:Z.append(C.bigram.scale)
	T=A.tied_embed_lr if A.tie_embeddings else A.embed_lr;A3=[{_H:[C.tok_emb.weight],_F:T,R:T}]
	if C.bigram is not _A:
		A3.append({_H:[C.bigram.embed.weight],_F:T,R:T})
		if C.bigram.proj is not _A:A2.append(C.bigram.proj.weight)
	AS=torch.optim.AdamW(A3,betas=(A.beta1,A.beta2),eps=A.adam_eps,weight_decay=A.weight_decay,fused=_B);h=Muon(A2,lr=A.matrix_lr,momentum=A.muon_momentum,backend_steps=A.muon_backend_steps,weight_decay=A.muon_wd)
	for O in h.param_groups:O[R]=A.matrix_lr
	AT=torch.optim.AdamW([{_H:Z,_F:A.scalar_lr,R:A.scalar_lr}],betas=(A.beta1,A.beta2),eps=A.adam_eps,weight_decay=A.weight_decay,fused=_B);L=[AS,h,AT]
	if C.lm_head is not _A:AU=torch.optim.Adam([{_H:[C.lm_head.weight],_F:A.head_lr,R:A.head_lr}],betas=(A.beta1,A.beta2),eps=A.adam_eps,fused=_B);L.insert(1,AU)
	AV=sum(A.numel()for A in C.parameters());B(f"model_params:{AV}");B(f"world_size:{E} grad_accum_steps:{G}");B(f"attention_mode:gqa num_heads:{A.num_heads} num_kv_heads:{A.num_kv_heads}");B(f"tie_embeddings:{A.tie_embeddings} embed_lr:{T} matrix_lr:{A.matrix_lr} scalar_lr:{A.scalar_lr}");B(f"train_batch_tokens:{A.train_batch_tokens} train_seq_len:{A.train_seq_len} iterations:{A.iterations} warmup_steps:{A.warmup_steps} max_wallclock_seconds:{A.max_wallclock_seconds:.3f}");B(f"seed:{A.seed}");i=DistributedTokenLoader(A.train_files,J,E,F)
	def U():
		for A in L:A.zero_grad(set_to_none=_B)
	V=1e3*A.max_wallclock_seconds if A.max_wallclock_seconds>0 else _A
	def AW(step,elapsed_ms):
		C=elapsed_ms;B=step
		if A.warmdown_iters<=0:return _D
		if V is _A:F=max(A.iterations-A.warmdown_iters,0);return max((A.iterations-B)/max(A.warmdown_iters,1),_E)if F<=B<A.iterations else _D
		G=C/max(B,1);D=A.warmdown_iters*G;E=max(V-C,_E);return E/max(D,1e-09)if E<=D else _D
	if A.warmup_steps>0:
		AX={A:B.detach().cpu().clone()for(A,B)in C.state_dict().items()};AY=[copy.deepcopy(A.state_dict())for A in L];I.train()
		for j in range(A.warmup_steps):
			U()
			for k in range(G):
				if H:I.require_backward_grad_sync=k==G-1
				l,m=i.next_batch(A.train_batch_tokens,A.train_seq_len,G)
				with torch.autocast(device_type=_G,dtype=torch.bfloat16,enabled=_B):AZ=I(l,m)
				(AZ*x).backward()
			for M in L:M.step()
			U()
			if A.warmup_steps<=20 or(j+1)%10==0 or j+1==A.warmup_steps:B(f"warmup_step:{j+1}/{A.warmup_steps}")
		C.load_state_dict(AX,strict=_B)
		for(M,Aa)in zip(L,AY,strict=_B):M.load_state_dict(Aa)
		U()
		if H:I.require_backward_grad_sync=_B
		i=DistributedTokenLoader(A.train_files,J,E,F)
	P=_E;Q=_A;W=_A;X=0;torch.cuda.synchronize();a=time.perf_counter();D=0
	while _B:
		A4=D==A.iterations or Q is not _A and D>=Q;Ab=A4 or A.val_loss_every>0 and D%A.val_loss_every==0
		if Ab:torch.cuda.synchronize();P+=1e3*(time.perf_counter()-a);Ac,Ad=eval_val(A,I,J,E,F,G,S,e,f,g);B(f"step:{D}/{A.iterations} val_loss:{Ac:.4f} val_bpb:{Ad:.4f} train_time:{P:.0f}ms step_avg:{P/max(D,1):.2f}ms");torch.cuda.synchronize();a=time.perf_counter()
		if A4:
			if Q is not _A and D<A.iterations:B(f"stopping_early: wallclock_cap train_time:{P:.0f}ms step:{D}/{A.iterations}")
			break
		Ae=P+1e3*(time.perf_counter()-a);A5=AW(D,Ae);U();n=torch.zeros((),device=F)
		for k in range(G):
			if H:I.require_backward_grad_sync=k==G-1
			l,m=i.next_batch(A.train_batch_tokens,A.train_seq_len,G)
			with torch.autocast(device_type=_G,dtype=torch.bfloat16,enabled=_B):A6=I(l,m)
			n+=A6.detach();(A6*x).backward()
		n/=G;A7=min(D/A.muon_momentum_warmup_steps,_D)if A.muon_momentum_warmup_steps>0 else _D;Af=(1-A7)*A.muon_momentum_warmup_start+A7*A.muon_momentum
		for O in h.param_groups:O[_O]=Af
		for M in L:
			for O in M.param_groups:O[_F]=O[R]*A5
		if A.grad_clip_norm>0:torch.nn.utils.clip_grad_norm_(C.parameters(),A.grad_clip_norm)
		for M in L:M.step()
		U();D+=1;o=P+1e3*(time.perf_counter()-a)
		if A.swa_enabled and A5<A.swa_start_frac and D%A.swa_every==0:
			if W is _A:W={A:B.detach().cpu().clone()for(A,B)in C.state_dict().items()};X=1;B(f"swa:start step:{D}")
			else:
				for(p,Ag)in C.state_dict().items():W[p]+=Ag.detach().cpu()
				X+=1
		Ah=A.train_log_every>0 and(D<=10 or D%A.train_log_every==0 or Q is not _A)
		if Ah:B(f"step:{D}/{A.iterations} train_loss:{n.item():.4f} train_time:{o:.0f}ms step_avg:{o/D:.2f}ms")
		q=V is not _A and o>=V
		if H and V is not _A:A8=torch.tensor(int(q),device=F);dist.all_reduce(A8,op=dist.ReduceOp.MAX);q=bool(A8.item())
		if Q is _A and q:Q=D
	B(f"peak memory allocated: {torch.cuda.max_memory_allocated()//1024//1024} MiB reserved: {torch.cuda.max_memory_reserved()//1024//1024} MiB")
	if A.swa_enabled and W is not _A and X>1:B(f"swa:applying averaged {X} checkpoints");Ai=C.state_dict();Aj={A:(B/X).to(dtype=Ai[A].dtype)for(A,B)in W.items()};C.load_state_dict(Aj,strict=_B)
	if K:torch.save(C.state_dict(),AM);A9=os.path.getsize(AM);b=len(c.encode(_I));B(f"Serialized model: {A9} bytes");B(f"Code size: {b} bytes");B(f"Total submission size: {A9+b} bytes")
	AA=A.prune_frac
	if AA>0:
		with torch.no_grad():
			for(p,N)in C.named_parameters():
				if N.ndim==2 and N.numel()>65536:Ak=torch.quantile(N.abs().float().flatten(),AA);AB=N.abs()<Ak;N.masked_fill_(AB,_E);AC=AB.sum().item();B(f"prune:{p} zeroed {AC}/{N.numel()} ({100*AC/N.numel():.1f}%)")
	AD={A:B.detach().cpu()for(A,B)in C.state_dict().items()};Al,Am=mixed_quantize_int6(AD,{_J,'attn','other'});AE=io.BytesIO();torch.save({'w':Al,'m':Am},AE);AF=AE.getvalue()
	if _COMPRESSOR==_K:AG=zstandard.ZstdCompressor(level=22).compress(AF)
	else:AG=zlib.compress(AF,9)
	if K:
		with open(v,'wb')as r:r.write(AG)
		AH=os.path.getsize(v);b=len(c.encode(_I));B(f"Serialized model int6+{_COMPRESSOR}: {AH} bytes");B(f"Total submission size int8+zlib: {AH+b} bytes")
	if H:dist.barrier()
	with open(v,'rb')as r:AI=r.read()
	if _COMPRESSOR==_K:AJ=zstandard.ZstdDecompressor().decompress(AI)
	else:AJ=zlib.decompress(AI)
	AK=torch.load(io.BytesIO(AJ),map_location='cpu');An=dequantize_mixed_int6(AK['w'],AK['m'],AD);C.load_state_dict(An,strict=_B)
	if A.ttt_enabled:
		if K:B(f"ttt:start lr={A.ttt_lr} momentum={A.ttt_momentum} epochs={A.ttt_epochs}")
		restore_low_dim_params_to_fp32(C)
		for s in C.parameters():s.requires_grad_(_B)
		ttt_adapt(A,C,F,S,rank=J,world_size=E,log_fn=B if K else _A)
		for s in C.parameters():s.requires_grad_(_C)
		if H:dist.barrier()
		if K:B('TTT complete, starting sliding window eval...')
	torch.cuda.synchronize();Ao=time.perf_counter()
	if A.eval_stride>0 and A.eval_stride<A.train_seq_len:B(f"final_eval_mode:sliding_window stride:{A.eval_stride} batch_seqs:{A.eval_batch_seqs}");t,u=eval_val_sliding(A,C,J,E,F,S,e,f,g,stride=A.eval_stride,batch_seqs=A.eval_batch_seqs)
	else:B('final_eval_mode:standard');t,u=eval_val(A,I,J,E,F,G,S,e,f,g)
	torch.cuda.synchronize();B(f"final_int8_zlib_roundtrip val_loss:{t:.4f} val_bpb:{u:.4f} eval_time:{1e3*(time.perf_counter()-Ao):.0f}ms");B(f"final_int8_zlib_roundtrip_exact val_loss:{t:.8f} val_bpb:{u:.8f}")
	if H:dist.destroy_process_group()
if __name__=='__main__':main()