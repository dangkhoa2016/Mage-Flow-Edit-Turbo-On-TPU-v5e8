#!/usr/bin/env python3
import argparse, hashlib, json, resource, sys, time
from pathlib import Path
import numpy as np
import jax, jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

def parse_topology(value):
    r,m=(int(x) for x in value.lower().split("x",1))
    if r*m!=8 or (r,m) not in {(1,8),(2,4),(4,2)}:
        raise ValueError("supported topologies: 1x8, 2x4, 4x2")
    return r,m

def main():
    ap=argparse.ArgumentParser()
    for k in ("runtime-root","checkpoint","manifest","input-dir","output"):
        ap.add_argument("--"+k,required=True)
    ap.add_argument("--topology",default="4x2")
    ap.add_argument("--resolution",type=int,choices=(512,768,1024),default=512)
    ap.add_argument("--seeds",required=True)
    args=ap.parse_args()
    replicas,model_axis=parse_topology(args.topology)
    seeds=[int(x) for x in args.seeds.split(",") if x.strip()]
    if len(seeds)!=replicas: raise SystemExit("seed count must equal replica count")
    sys.path.insert(0,args.runtime_root)
    from experiment import image_postprocess, vae_driver
    if jax.default_backend()!="tpu" or len(jax.devices())!=8: raise SystemExit("TPU/8 required")
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    mesh=Mesh(np.asarray(jax.devices()).reshape(replicas,model_axis),("replica","model"))
    rt=vae_driver.restore_vae(vae_checkpoint=args.checkpoint,manifest_path=args.manifest,execute_compute=True)
    if not rt.bound or rt.params is None or len(rt.params)!=728: raise SystemExit("VAE binding failed")
    rep=NamedSharding(mesh,P())
    rt.params={k:jax.device_put(jnp.asarray(v),rep) for k,v in rt.params.items()}
    H=W=args.resolution//16
    host=np.concatenate([np.load(Path(args.input_dir)/f"C5_seed_{s}.npy") for s in seeds],axis=0)
    if host.shape!=(replicas,128,H,W): raise SystemExit(f"bad latent shape {host.shape}")
    latent=jax.device_put(jnp.asarray(host,dtype=jnp.bfloat16),NamedSharding(mesh,P("replica",None,None,None)))
    def decode_once():
        t=time.time()
        y=vae_driver.decode_latent(rt,latent,hooks=None,execute_compute=True)["tensor"]
        jax.block_until_ready(y)
        return np.asarray(jax.device_get(y)),time.time()-t
    cold,cold_s=decode_once()
    expected=(replicas,3,args.resolution,args.resolution)
    if cold.shape!=expected or not np.isfinite(cold).all(): raise SystemExit("VAE cold gate failed")
    warm,warm_s=decode_once()
    if not np.array_equal(cold,warm): raise SystemExit("VAE warm rerun is not bit-exact")
    rows=[]
    for i,seed in enumerate(seeds):
        raw=cold[i:i+1]
        np.save(out/f"C6_seed_{seed}.npy",raw)
        u8=image_postprocess.to_uint8_image(raw)
        png=image_postprocess.encode_png(u8)
        sha=hashlib.sha256(png).hexdigest()
        (out/f"seed_{seed}.png").write_bytes(png)
        rows.append({"seed":seed,"shape":list(raw.shape),"dtype":str(raw.dtype),
                     "min":float(raw.min()),"max":float(raw.max()),
                     "mean":float(raw.mean()),"std":float(raw.std()),
                     "png_sha256":sha,"pixel_stats":image_postprocess.pixel_stats(u8)})
    if len({r["png_sha256"] for r in rows})!=replicas: raise SystemExit("PNG outputs are not unique")
    hbm=[]
    for i,d in enumerate(jax.devices()):
        m=d.memory_stats() or {}
        hbm.append({"device":i,"peak_bytes_in_use":m.get("peak_bytes_in_use"),
                    "bytes_in_use":m.get("bytes_in_use"),"bytes_limit":m.get("bytes_limit")})
    summary={"status":"PASS","stage":"vae","backend":jax.default_backend(),"device_count":8,
      "topology":args.topology,"resolution":[args.resolution,args.resolution],
      "seeds":seeds,"binding_coverage":"728/728",
      "cold_wall_s":cold_s,"warm_wall_s":warm_s,
      "warm_effective_s_per_image":warm_s/replicas,
      "warm_images_per_min":60.0*replicas/warm_s,
      "deterministic_bit_exact":True,"outputs_unique":True,"outputs":rows,
      "host_maxrss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,"hbm":hbm}
    (out/"vae_stage_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2),flush=True)

if __name__=="__main__":
    main()
