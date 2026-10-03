#!/usr/bin/env python3
import argparse, json, os, subprocess, sys, time
from pathlib import Path

BOOTSTRAP=Path(__file__).resolve().parent
BASE=BOOTSTRAP.parent
RUNTIME=BASE/'runtime'

ap=argparse.ArgumentParser(description='Mage-Flow Edit-Turbo full multimodal JAX/TPU runner')
ap.add_argument('--image',required=True,help='Reference image')
ap.add_argument('--prompt',required=True,help='Edit instruction')
ap.add_argument('--output',required=True,help='Output directory')
ap.add_argument('--resolution',type=int,choices=(512,768,1024),default=512)
ap.add_argument('--topology',choices=('4x2',),default='4x2')
ap.add_argument('--runtime-site',default='/kaggle/working/mage-flow-tpu-runtime-site')
ap.add_argument('--artifact-root',required=True,help='External model artifact root containing text_encoder/checkpoints/manifests/rope')
ap.add_argument('--seeds',default='42,43,44,45')
args=ap.parse_args()
ARTIFACT=Path(args.artifact_root).resolve()
if not ARTIFACT.exists(): raise SystemExit(f'artifact root does not exist: {ARTIFACT}')

out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
cond_dir=out/'conditioning'; ref_dir=out/'reference'; tf_dir=out/'transformer'; vae_dir=out/'vae'
for d in (cond_dir,ref_dir,tf_dir,vae_dir): d.mkdir(exist_ok=True)
cond=cond_dir/'C0_edit_conditioning_f32.npy'
ref=ref_dir/'reference_latent.npy'
env=os.environ.copy()
env.update({
    'PJRT_DEVICE':'TPU','TPU_ACCELERATOR_TYPE':'v5litepod-8',
    'TPU_CHIPS_PER_HOST_BOUNDS':'2,4,1','TPU_HOST_BOUNDS':'1,1,1',
    'TPU_PROCESS_ADDRESSES':'local','TPU_SKIP_MDS_QUERY':'1',
    'TPU_WORKER_HOSTNAMES':'localhost','TPU_WORKER_ID':'0','KERAS_BACKEND':'jax',
    'USE_TORCH_XLA':'0',
})
env['PYTHONPATH']=':'.join([args.runtime_site,str(RUNTIME),env.get('PYTHONPATH','')])
py=sys.executable

def run(name,cmd):
    print(f'=== {name} ===',flush=True)
    t=time.time()
    subprocess.run(cmd,check=True,env=env)
    return time.time()-t

timings={}
timings['conditioning']=run('MULTIMODAL CONDITIONING',[
    py,str(BOOTSTRAP/'08_tpu_stage_multimodal_conditioning.py'),
    '--runtime-root',str(RUNTIME),'--checkpoint',str(ARTIFACT/'checkpoints/text_encoder'),
    '--model-root',str(ARTIFACT),'--processor-root',str(BASE/'qwen3vl-processor-assets'),
    '--image',args.image,'--prompt',args.prompt,'--output',str(cond),
    '--resolution',str(args.resolution),
])

timings['reference_vae']=run('REFERENCE VAE ENCODE',[
    py,str(BOOTSTRAP/'08_tpu_stage_reference_vae.py'),
    '--runtime-root',str(RUNTIME),'--checkpoint',str(ARTIFACT/'checkpoints/vae'),
    '--manifest',str(ARTIFACT/'manifests/VAE_RUNTIME_BINDING_MANIFEST.json'),
    '--image',args.image,'--output',str(ref),'--resolution',str(args.resolution),
])

timings['transformer']=run('EDIT TRANSFORMER',[
    py,str(BOOTSTRAP/'08_tpu_stage_edit_transformer.py'),
    '--runtime-root',str(RUNTIME),'--source-root',str(RUNTIME/'source'),
    '--checkpoint',str(ARTIFACT/'checkpoints/transformer'),
    '--basis-dim16',str(ARTIFACT/'rope/captured-basis-dim16.npy'),
    '--basis-dim56',str(ARTIFACT/'rope/captured-basis-dim56.npy'),
    '--conditioning',str(cond),'--reference-latent',str(ref),'--output',str(tf_dir),
    '--resolution',str(args.resolution),'--seeds',args.seeds,
])
timings['vae_decode']=run('VAE DECODE',[
    py,str(BOOTSTRAP/'06_tpu_stage_vae.py'),
    '--runtime-root',str(RUNTIME),'--checkpoint',str(ARTIFACT/'checkpoints/vae'),
    '--manifest',str(ARTIFACT/'manifests/VAE_RUNTIME_BINDING_MANIFEST.json'),
    '--input-dir',str(tf_dir),'--output',str(vae_dir),'--topology',args.topology,
    '--resolution',str(args.resolution),'--seeds',args.seeds,
])

seed_list=[int(x) for x in args.seeds.split(',') if x.strip()]
pngs=[str(vae_dir/f'seed_{s}.png') for s in seed_list]
for p in pngs:
    if not Path(p).is_file(): raise SystemExit(f'missing output {p}')
summary={
    'status':'PASS','runner':'bootstrap/08_run_tpu_edit.py','image':args.image,'prompt':args.prompt,
    'resolution':[args.resolution,args.resolution],'topology':args.topology,'seeds':seed_list,
    'stage_wall_s':timings,'pngs':pngs,
}
(out/'generation_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2),flush=True)
