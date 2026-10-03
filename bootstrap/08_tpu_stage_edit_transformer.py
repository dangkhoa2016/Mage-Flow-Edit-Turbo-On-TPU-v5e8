import argparse, json, resource, sys, time
from pathlib import Path
import numpy as np
import jax, jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

ap=argparse.ArgumentParser()
for k in ('runtime-root','source-root','checkpoint','basis-dim16','basis-dim56','conditioning','reference-latent','output'):
    ap.add_argument('--'+k, required=True)
ap.add_argument('--resolution',type=int,default=512)
ap.add_argument('--seeds',default='42,43,44,45')
args=ap.parse_args(); sys.path.insert(0,args.runtime_root)
from experiment import rope_provider, transformer_driver
from experiment.runtime_contract import SIGMA_SCHEDULE
if jax.default_backend()!='tpu' or len(jax.devices())!=8: raise SystemExit('TPU/8 required')
R,M=4,2; C=128; H=W=args.resolution//16; n=H*W
seeds=[int(x) for x in args.seeds.split(',')]; assert len(seeds)==R
mesh=Mesh(np.asarray(jax.devices()).reshape(R,M),('replica','model')); cpu=jax.devices('cpu')[0]
cond=np.load(args.conditioning).astype(np.float32); T=cond.shape[1]
txt=jax.device_put(np.concatenate([cond]*R,axis=1),NamedSharding(mesh,P(None,'replica',None)))
ref=np.load(args.reference_latent).astype(np.float32)
if ref.shape!=(1,C,H,W): raise SystemExit(f'bad reference latent {ref.shape}')
ref=np.concatenate([ref]*R,axis=0)
ref=jax.device_put(jnp.asarray(ref),NamedSharding(mesh,P('replica',None,None,None)))
with jax.default_device(cpu):
    source=transformer_driver.load_transformer_source(args.source_root)
    model=transformer_driver.construct_transformer_model(source)
    restored=transformer_driver.restore_transformer_checkpoint(args.checkpoint)
paths=model.parameter_paths(); plan=transformer_driver.static_sharding_plan(model,mesh_size=M,expect_contract=True)
for key,var in paths.items():
    host=np.asarray(jax.device_get(restored[key])).astype(np.float32,copy=False)
    var.assign(jax.device_put(host,NamedSharding(mesh,P(*plan['plan'][key]))))
del restored
img_cu=jax.device_put(np.arange(R+1,dtype=np.int32)*(2*n),NamedSharding(mesh,P()))
txt_cu=jax.device_put(np.arange(R+1,dtype=np.int32)*T,NamedSharding(mesh,P()))
rope_one=rope_provider.build_image_rope_sequence('control',[(1,H,W),(1,H,W)],args.basis_dim16,args.basis_dim56)
if rope_one.shape!=(2*n,64,2): raise SystemExit(f'bad edit rope {rope_one.shape}')
rope=jax.device_put(np.concatenate([rope_one]*R,axis=0),NamedSharding(mesh,P('replica',None,None)))
latent_sh=NamedSharding(mesh,P('replica',None,None,None)); token_sh=NamedSharding(mesh,P(None,'replica',None)); time_sh=NamedSharding(mesh,P('replica'))
xs=[jax.random.normal(jax.random.PRNGKey(seed),(1,C,H,W),dtype=jnp.float32) for seed in seeds]
x=jax.device_put(jnp.concatenate(xs,axis=0),latent_sh)
step_s=[]; wall=time.time()
for i in range(4):
    sigma=float(SIGMA_SCHEDULE[i]); sigma_next=float(SIGMA_SCHEDULE[i+1])
    target=jnp.transpose(x,(0,2,3,1)).reshape(R,n,C)
    reference=jnp.transpose(ref,(0,2,3,1)).reshape(R,n,C)
    packed=jnp.stack([target,reference],axis=1).reshape(1,R*2*n,C)
    packed=jax.device_put(packed,token_sh)
    ts=jax.device_put(np.full((R,),sigma,np.float32),time_sh)
    t=time.time(); pred=model.forward(packed,txt,ts,rope,img_cu,txt_cu,joint_attention_kwargs={'attention_mode':'segmented-query-chunk','query_chunk':256})
    pred=pred.reshape(R,2,n,C)[:,0]
    velocity=jnp.transpose(pred.reshape(R,H,W,C),(0,3,1,2))
    x=jax.device_put(x+np.float32(sigma_next-sigma)*velocity,latent_sh)
    jax.block_until_ready(x); step_s.append(time.time()-t)
out_arr=np.asarray(jax.device_get(x)); out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
for i,seed in enumerate(seeds): np.save(out/f'C5_seed_{seed}.npy',out_arr[i:i+1])
hbm=[]
for i,d in enumerate(jax.devices()):
    m=d.memory_stats() or {}; hbm.append({'device':i,'peak_bytes_in_use':m.get('peak_bytes_in_use'),'bytes_limit':m.get('bytes_limit')})
summary={'status':'PASS','stage':'edit-transformer-multimodal','visual_conditioning':'QWEN3VL_JAX_POOLER_PLUS_DEEPSTACK','reference_latent':True,'layout':'per-replica[target,reference]','resolution':[args.resolution,args.resolution],'seeds':seeds,'conditioning_tokens':T,'image_tokens_per_replica':2*n,'steps':4,'wall_s':time.time()-wall,'step_s':step_s,'parameter_leaves':len(paths),'parameter_sharded':plan['sharded'],'parameter_replicated':plan['replicated'],'host_maxrss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'hbm':hbm}
(out/'multimodal_edit_summary.json').write_text(json.dumps(summary,indent=2)+'\n'); print(json.dumps(summary,indent=2))
