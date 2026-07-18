# Sailor scan-pipeline cloud handler — Lane B (cloud-v1), RunPod serverless.
# Same job spec as the local lane (tools/scan-pipeline/worker.mjs): frames
# (ffmpeg) → COLMAP (features/match/map, best-submodel selection) → OpenSplat
# (CUDA, 30k iters) → standard 3DGS .ply. Finishing (ply2spz, calibration,
# manifest) stays on the Mac — single-sourced.
#
# Transport is RunPod-native: the dispatcher moves files as chunked base64
# through the job API onto the network volume (no S3 keys, no third service).
# Ops: put (write chunk) · run (full pipeline) · stat · fetch (read chunk) · rm.
#
# The Mac scars do NOT apply here by construction: COLMAP is pinned 3.11.1
# (3.x bin layout OpenSplat parses natively — no colmap4-to-3), and Linux
# has a single OpenMP runtime (no KMP_DUPLICATE_LIB_OK pin).
import base64
import gzip
import json
import os
import shutil
import subprocess
import time

import runpod

VOL = "/runpod-volume/sailor-scans"


def safe(rel):
    p = os.path.normpath(rel).lstrip("/")
    if p.startswith(".."):
        raise ValueError("path escapes the volume")
    return p


def vol(rel):
    return os.path.join(VOL, safe(rel))


def sh(cmd, cwd=None, env=None):
    t0 = time.time()
    p = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(
            f"{os.path.basename(cmd[0])} exited {p.returncode} :: {p.stderr[-500:] or p.stdout[-500:]}"
        )
    return time.time() - t0, p.stdout, p.stderr


def op_put(inp):
    full = vol(inp["path"])
    os.makedirs(os.path.dirname(full), exist_ok=True)
    data = base64.b64decode(inp["b64"])
    with open(full, "ab" if inp.get("append") else "wb") as f:
        f.write(data)
    return {"bytes": os.path.getsize(full)}


def op_stat(inp):
    full = vol(inp["path"])
    if not os.path.exists(full):
        return {"exists": False}
    return {"exists": True, "bytes": os.path.getsize(full)}


def op_fetch(inp):
    full = vol(inp["path"])
    size = os.path.getsize(full)
    off = int(inp.get("offset", 0))
    length = int(inp.get("length", 6_000_000))
    with open(full, "rb") as f:
        f.seek(off)
        data = f.read(length)
    return {
        "b64": base64.b64encode(data).decode(),
        "offset": off,
        "bytes": len(data),
        "total": size,
        "eof": off + len(data) >= size,
    }


def op_rm(inp):
    full = vol(inp["path"])
    if os.path.isdir(full):
        shutil.rmtree(full, ignore_errors=True)
    elif os.path.exists(full):
        os.remove(full)
    return {"removed": True}


def op_run(inp):
    job_id = safe(inp["jobId"])
    params = inp.get("params", {})
    fps = params.get("fps", 10)
    long_side = params.get("longSide", 2160)
    max_frames = params.get("maxFrames", 260)
    iters = params.get("iters", 30000)
    videos = inp["videos"]  # names under <jobId>/in/, split as <name>.partNNN

    work = f"/tmp/{job_id}"
    shutil.rmtree(work, ignore_errors=True)
    frames = os.path.join(work, "frames")
    os.makedirs(frames, exist_ok=True)
    stats = {"steps": {}, "params": {"fps": fps, "longSide": long_side, "maxFrames": max_frames, "iters": iters}}

    # 0 — reassemble the chunked uploads from the volume
    updir = os.path.join(work, "uploads")
    os.makedirs(updir, exist_ok=True)
    for name in videos:
        indir = vol(f"{job_id}/in")
        parts = sorted(p for p in os.listdir(indir) if p.startswith(name + ".part"))
        if not parts:
            raise RuntimeError(f"no uploaded parts for {name}")
        with open(os.path.join(updir, name), "wb") as out:
            for p in parts:
                with open(os.path.join(indir, p), "rb") as f:
                    shutil.copyfileobj(f, out)

    # 1 — frames (same filter as the local lane, long side parameterized)
    scale = f"scale='if(gt(iw,ih),{long_side},-2)':'if(gt(iw,ih),-2,{long_side})'"
    fi = 0
    for name in videos:
        secs, _, _ = sh([
            "ffmpeg", "-i", os.path.join(updir, name),
            "-vf", f"fps={fps},{scale}",
            "-q:v", "2",
            "-start_number", str(fi * 1000),
            os.path.join(frames, "f%05d.jpg"),
        ])
        fi += 1
    frame_list = sorted(f for f in os.listdir(frames) if f.endswith(".jpg"))
    if len(frame_list) > max_frames:  # evenly thin, exactly like worker.mjs
        keep = {frame_list[(i * len(frame_list)) // max_frames] for i in range(max_frames)}
        for f in frame_list:
            if f not in keep:
                os.remove(os.path.join(frames, f))
        frame_list = sorted(keep)
    stats["steps"]["frames"] = {"count": len(frame_list)}
    if len(frame_list) < 20:
        raise RuntimeError(f"only {len(frame_list)} frames extracted — footage too short")

    # 2 — COLMAP (CUDA SIFT), same camera model + matcher as the local lane.
    # NOTE: COLMAP 3.x spells these Sift{Extraction,Matching}.use_gpu — the
    # Feature{Extraction,Matching}.* names in worker.mjs are 4.x renames.
    db = os.path.join(work, "db.db")
    secs, _, _ = sh(["colmap", "feature_extractor", "--database_path", db, "--image_path", frames,
                     "--ImageReader.camera_model", "SIMPLE_RADIAL", "--ImageReader.single_camera", "1",
                     "--SiftExtraction.use_gpu", "1"])
    stats["steps"]["features"] = {"secs": round(secs)}
    secs, _, _ = sh(["colmap", "sequential_matcher", "--database_path", db,
                     "--SiftMatching.use_gpu", "1", "--SequentialMatching.overlap", "25"])
    stats["steps"]["match"] = {"secs": round(secs)}
    sparse = os.path.join(work, "sparse")
    os.makedirs(sparse, exist_ok=True)
    secs, _, _ = sh(["colmap", "mapper", "--database_path", db, "--image_path", frames,
                     "--output_path", sparse])
    stats["steps"]["map"] = {"secs": round(secs)}
    models = [m for m in os.listdir(sparse) if m.isdigit()]
    if "0" not in models:
        raise RuntimeError("COLMAP could not register the cameras — footage may lack overlap/parallax")
    # best-submodel selection, ported from worker.mjs (jump cuts fragment maps)
    best, best_n = "0", -1
    for m in models:
        try:
            _, out, err = sh(["colmap", "model_analyzer", "--path", os.path.join(sparse, m)])
        except RuntimeError:
            continue
        txt = out + err
        n = 0
        for line in txt.splitlines():
            if "Registered images" in line:
                n = int(line.split(":")[1].strip())
        if n > best_n:
            best, best_n = m, n
    stats["steps"]["map"]["subModels"] = len(models)
    stats["steps"]["map"]["bestModelImages"] = best_n
    if best != "0":
        os.rename(os.path.join(sparse, "0"), os.path.join(sparse, "_frag0"))
        os.rename(os.path.join(sparse, best), os.path.join(sparse, "0"))

    txt_dir = os.path.join(work, "sparse_txt")
    os.makedirs(txt_dir, exist_ok=True)
    sh(["colmap", "model_converter", "--input_path", os.path.join(sparse, "0"),
        "--output_path", txt_dir, "--output_type", "TXT"])
    n_reg = 0
    with open(os.path.join(txt_dir, "images.txt")) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 10 and parts[9].lower().endswith((".jpg", ".jpeg", ".png")):
                n_reg += 1
    stats["steps"]["cameras"] = {"registered": n_reg}

    # 3 — OpenSplat (CUDA). Relative project path + cwd, mirroring the local
    # invocation. COLMAP 3.x bins are read natively here.
    images_link = os.path.join(work, "images")
    if not os.path.exists(images_link):
        os.symlink("frames", images_link)
    ply = os.path.join(work, "splat.ply")
    secs, _, _ = sh(["opensplat", ".", "-n", str(iters), "-o", ply], cwd=work)
    stats["steps"]["train"] = {"secs": round(secs)}
    if not os.path.exists(ply):
        raise RuntimeError("OpenSplat produced no output")
    stats["plyBytes"] = os.path.getsize(ply)

    # 4 — land outputs on the volume for chunked fetch
    outdir = vol(f"{job_id}/out")
    os.makedirs(outdir, exist_ok=True)
    shutil.copyfile(ply, os.path.join(outdir, "splat.ply"))
    with open(os.path.join(txt_dir, "images.txt"), "rb") as f_in, gzip.open(
        os.path.join(outdir, "images.txt.gz"), "wb"
    ) as f_out:
        shutil.copyfileobj(f_in, f_out)
    with open(os.path.join(outdir, "stats.json"), "w") as f:
        json.dump(stats, f)
    shutil.rmtree(work, ignore_errors=True)
    return stats


OPS = {"put": op_put, "stat": op_stat, "fetch": op_fetch, "rm": op_rm, "run": op_run}


def handler(job):
    inp = job.get("input") or {}
    op = inp.get("op")
    if op not in OPS:
        return {"error": f"unknown op: {op}"}
    try:
        return OPS[op](inp)
    except Exception as e:  # honest failure back to the dispatcher
        return {"error": str(e)[:800]}


runpod.serverless.start({"handler": handler})
