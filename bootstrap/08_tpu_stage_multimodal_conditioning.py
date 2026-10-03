#!/usr/bin/env python3
import os
os.environ.setdefault('USE_TORCH_XLA','0')
import argparse, json, sys, time
from pathlib import Path
import numpy as np
from PIL import Image
from transformers import AutoProcessor

ap=argparse.ArgumentParser(description='Qwen3-VL multimodal conditioning for Mage-Flow Edit')
for k in ('runtime-root','checkpoint','model-root','processor-root','image','prompt','output'):
    ap.add_argument('--'+k,required=True)
ap.add_argument('--resolution',type=int,choices=(512,768,1024),default=512)
args=ap.parse_args()
sys.path.insert(0,args.runtime_root)

TEMPLATE=("<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), "
          "then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements "
          "while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n"
          "<|vision_start|><|image_pad|><|vision_end|>{}<|im_end|>\n<|im_start|>assistant\n")
DROP_IDX=64
from experiment.text_encoder_runtime import restore_text_encoder_checkpoint, config_from_model_root
from experiment import qwen3vl_vision_runtime as vr
from experiment import qwen3vl_multimodal_runtime as mm
import jax

if jax.default_backend()!='tpu' or len(jax.devices())!=8:
    raise SystemExit('TPU v5e-8 with 8 devices is required')

res=args.resolution
image=Image.open(args.image).convert('RGB').resize((res,res),Image.Resampling.LANCZOS)
processor=AutoProcessor.from_pretrained(args.processor_root,local_files_only=True)
text=TEMPLATE.format(args.prompt)
inputs=processor(text=[text],images=[image],padding=True,return_tensors='pt')
ids=inputs['input_ids'].numpy().astype(np.int32)
att=inputs['attention_mask'].numpy().astype(np.int32)
types=inputs['mm_token_type_ids'].numpy().astype(np.int32)
pixel=inputs['pixel_values'].numpy().astype(np.float32)
grid=inputs['image_grid_thw'].numpy().astype(np.int32)
pos=mm.multimodal_position_ids(ids,types,grid,2,att)
if int((ids==mm.IMAGE_TOKEN_ID).sum()) != int(np.prod(grid)//4):
    raise SystemExit('image token count/grid contract failed')
cfg=config_from_model_root(args.model_root)
print('restoring Qwen3-VL JAX checkpoint',flush=True)
t0=time.time(); state=restore_text_encoder_checkpoint(args.checkpoint)
visual={k:v for k,v in state.items() if k.startswith('text_encoder.visual.')}
lm={k:v for k,v in state.items() if k.startswith('text_encoder.language_model.')}
if len(visual)!=315 or len(lm)!=398:
    raise SystemExit(f'bad TE split visual={len(visual)} lm={len(lm)}')
del state

vision_fn=jax.jit(lambda p,x:vr.vision_forward(p,x,grid))
t1=time.time(); v=vision_fn(visual,pixel); jax.block_until_ready(v['pooler_output']); vision_s=time.time()-t1
lm_fn=jax.jit(lambda p,img,d0,d1,d2:mm.multimodal_language_forward(p,cfg,ids,pos,img,[d0,d1,d2]))
t2=time.time(); h=lm_fn(lm,v['pooler_output'],v['deepstack_features'][0],v['deepstack_features'][1],v['deepstack_features'][2]); jax.block_until_ready(h); lm_s=time.time()-t2
host=np.asarray(jax.device_get(h),np.float32)
cond=host[:,DROP_IDX:,:]
if cond.ndim!=3 or cond.shape[0]!=1 or cond.shape[2]!=2560 or not np.isfinite(cond).all():
    raise SystemExit(f'bad conditioning {cond.shape}')
out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True); np.save(out,cond)
summary={
    'status':'PASS','stage':'qwen3vl-multimodal-conditioning','resolution':[res,res],
    'input_tokens':int(ids.shape[1]),'image_tokens':int((ids==mm.IMAGE_TOKEN_ID).sum()),
    'conditioning_tokens':int(cond.shape[1]),'conditioning_shape':list(cond.shape),
    'grid_thw':grid.tolist(),'vision_s':vision_s,'language_model_s':lm_s,
    'total_s':time.time()-t0,'drop_idx':DROP_IDX,
}
(out.parent/'multimodal_conditioning_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2),flush=True)
