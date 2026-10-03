import argparse, json, sys, time
from pathlib import Path
import numpy as np
from PIL import Image
import jax, jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

ap=argparse.ArgumentParser()
ap.add_argument('--runtime-root', required=True)
ap.add_argument('--checkpoint', required=True)
ap.add_argument('--manifest', required=True)
ap.add_argument('--image', required=True)
ap.add_argument('--output', required=True)
ap.add_argument('--resolution', type=int, default=512)
args=ap.parse_args()
sys.path.insert(0,args.runtime_root)
from experiment import vae_driver
if jax.default_backend()!='tpu' or len(jax.devices())!=8: raise SystemExit('TPU/8 required')
res=args.resolution
im=Image.open(args.image).convert('RGB').resize((res,res), Image.Resampling.LANCZOS)
host=np.asarray(im,dtype=np.float32)/127.5-1.0
host=host.transpose(2,0,1)[None,...]
mesh=Mesh(np.asarray(jax.devices()).reshape(4,2),('replica','model'))
rt=vae_driver.restore_vae(vae_checkpoint=args.checkpoint,manifest_path=args.manifest,execute_compute=True)
rep=NamedSharding(mesh,P())
rt.params={k:jax.device_put(jnp.asarray(v),rep) for k,v in rt.params.items()}
image=jax.device_put(jnp.asarray(host,dtype=jnp.bfloat16),NamedSharding(mesh,P(None,None,None,None)))
t=time.time(); latent=rt.encode_fn(rt.params,image) if getattr(rt,'encode_fn',None) else None
if latent is None:
    from experiment import vae_runtime
    latent=vae_runtime.encode(rt.params,image)
jax.block_until_ready(latent)
arr=np.asarray(jax.device_get(latent),dtype=np.float32)
if arr.shape!=(1,128,res//16,res//16): raise SystemExit(f'bad ref latent shape {arr.shape}')
out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True); np.save(out,arr)
print(json.dumps({'status':'PASS','shape':list(arr.shape),'min':float(arr.min()),'max':float(arr.max()),'mean':float(arr.mean()),'std':float(arr.std()),'elapsed_s':time.time()-t},indent=2))
