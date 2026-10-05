"""
prepare_benchmark.py
====================
Preparazione del benchmark paired CelebV-HQ (real) vs WAN 2.2 I2V (fake).

Passi:
  1. INDEX    - costruisce l'indice di tutti i video (405 real + 9 azioni x 405 fake)
                a partire dai CSV di metadati e verifica abbinamenti e file mancanti.
  2. EXTRACT  - per ogni video estrae i frame [start_frame, start_frame + num_frames)
                (default 1..72: il frame 0 del fake coincide con il first frame reale),
                li ridimensiona tutti alla stessa risoluzione con la stessa interpolazione
                e li salva su disco. Real e fake passano per la stessa identica pipeline.
  3. WINDOWS  - genera le sliding window (default lunghezza 16, stride 8) come righe di
                un indice: nessun frame viene duplicato su disco.
  4. SPLITS   - split per identita' (yt_id) e generazione dei protocolli:
                  P1  Identity Shift         : id di test non visti, tutte le azioni viste
                  P2  Action Shift           : azione non vista, identita' dei fake viste
                  P3  Identity+Action Shift  : azione non vista e identita' non viste
                per leave-one-action-out (9 fold) e leave-one-category-out (3 fold).
  5. QA       - report di confronto real vs fake (codec, fps, frame, risoluzione, bitrate).

Uso (dalla cartella che contiene final_dataset/ e generated/):
    pip install opencv-python pandas numpy
    python prepare_benchmark.py
(default: output in D:\benchmark, frame in JPEG qualita' 95)

Per cambiare solo finestre o split senza ri-estrarre i frame:
    python prepare_benchmark.py --skip-extract --win-len 32 --win-stride 16
"""

import argparse
import json
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------
# 1. INDEX
# ----------------------------------------------------------------------------
def build_index(root: Path) -> pd.DataFrame:
    real_dir = root / "final_dataset" / "final_dataset"
    gen_dir = root / "generated" / "generated"

    df_r = pd.read_csv(real_dir / "metadata_final.csv")
    reals = pd.DataFrame({
        "video_id": "real__" + df_r["clip_id"],
        "clip_id": df_r["clip_id"],
        "yt_id": df_r["yt_id"],
        "label": 0,
        "action": "real",
        "category": "real",
        "src_path": [str(real_dir / p) for p in df_r["processed_path"]],
    })

    rows = []
    # struttura: generated/generated/<categoria>/<azione>/metadata_*.csv
    # l'azione si prende dal nome della CARTELLA (es. head_turn), non dal nome file
    # (che usa "headturn"), cosi' i nomi sono coerenti.
    for meta in sorted(gen_dir.glob("*/*/metadata_*.csv")):
        action = meta.parent.name
        category = meta.parent.parent.name
        df = pd.read_csv(meta)
        for r in df.itertuples(index=False):
            rows.append({
                "video_id": f"{action}__{r.clip_id}",
                "clip_id": r.clip_id,
                "yt_id": r.yt_id,
                "label": 1,
                "action": action,
                "category": category,
                "src_path": str(meta.parent / "videos" / r.file_name),
            })
    fakes = pd.DataFrame(rows)
    if fakes.empty:
        raise SystemExit(f"Nessun metadata_*.csv trovato sotto {gen_dir}")

    idx = pd.concat([reals, fakes], ignore_index=True)

    # --- controlli -------------------------------------------------------
    print(f"[index] real: {len(reals)}  fake: {len(fakes)}  "
          f"azioni: {fakes.action.nunique()}  categorie: {fakes.category.nunique()}")
    print(fakes.groupby(["category", "action"]).size().to_string())

    missing = idx[~idx.src_path.map(os.path.exists)]
    if len(missing):
        print(f"[index] ATTENZIONE: {len(missing)} file mancanti (esempi):")
        print(missing.src_path.head().to_string())
        idx = idx.drop(missing.index).reset_index(drop=True)

    orphan = set(fakes.clip_id) - set(reals.clip_id)
    if orphan:
        print(f"[index] ATTENZIONE: {len(orphan)} fake senza real corrispondente, rimossi")
        idx = idx[~idx.clip_id.isin(orphan)].reset_index(drop=True)

    return idx


# ----------------------------------------------------------------------------
# 2. EXTRACT
# ----------------------------------------------------------------------------
def _fourcc(v: float) -> str:
    v = int(v)
    return "".join(chr((v >> 8 * i) & 0xFF) for i in range(4)).strip("\x00")


def extract_video(src: str, dst: str, start: int, num: int, size: int, ext: str) -> dict:
    """Legge i frame [start, start+num), li ridimensiona e li salva. Restituisce statistiche."""
    out = {"src_path": src, "ok": False}
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        out["error"] = "impossibile aprire il video"
        return out

    fps = cap.get(cv2.CAP_PROP_FPS)
    n_meta = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out.update(
        codec=_fourcc(cap.get(cv2.CAP_PROP_FOURCC)),
        fps=round(fps, 3),
        frame_count=n_meta,
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    dur = n_meta / fps if fps > 0 else 0
    out["bitrate_kbps"] = round(os.path.getsize(src) * 8 / dur / 1000, 1) if dur > 0 else None

    dst_dir = Path(dst)
    dst_dir.mkdir(parents=True, exist_ok=True)
    params = [cv2.IMWRITE_JPEG_QUALITY, 95] if ext == "jpg" else [cv2.IMWRITE_PNG_COMPRESSION, 3]

    i, saved = 0, 0
    while i < start + num:
        ok, frame = cap.read()
        if not ok:
            break
        if i >= start:
            frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
            # imencode + tofile: funziona anche con percorsi Windows non-ASCII
            # e, a differenza di cv2.imwrite, solleva un errore esplicito se la scrittura fallisce
            ok_enc, buf = cv2.imencode(f".{ext}", frame, params)
            if not ok_enc:
                out["error"] = f"codifica fallita al frame {i}"
                break
            try:
                buf.tofile(str(dst_dir / f"{i:03d}.{ext}"))
            except OSError as e:
                out["error"] = f"scrittura fallita al frame {i}: {e}"
                break
            saved += 1
        i += 1
    cap.release()

    out["frames_saved"] = saved
    out["ok"] = saved == num
    if not out["ok"] and "error" not in out:
        out["error"] = f"salvati {saved}/{num} frame (video troppo corto?)"
    return out


def check_disk(idx: pd.DataFrame, out: Path, args):
    """Stima lo spazio necessario e si ferma prima di iniziare se non basta."""
    import shutil
    bpp = 0.15 if args.ext == "jpg" else 1.6   # byte per pixel, stima prudente
    need = len(idx) * args.num_frames * args.size * args.size * bpp
    free = shutil.disk_usage(out.resolve()).free
    print(f"[disk] spazio stimato: {need / 1e9:.1f} GB  -  libero su {out.resolve().anchor}: {free / 1e9:.1f} GB")
    if need > free * 0.95 and not args.force:
        raise SystemExit(
            "[disk] Spazio insufficiente. Opzioni: --ext jpg, --size 224, "
            "--out su un altro disco (es. --out D:\\benchmark), oppure --force per procedere comunque.")


def run_extract(idx: pd.DataFrame, out: Path, args) -> pd.DataFrame:
    frames_root = out / "frames"
    idx = idx.copy()
    idx["frames_dir"] = [
        str(Path("frames") / a / c) for a, c in zip(idx.action, idx.clip_id)
    ]

    check_disk(idx, out, args)

    stats = []
    n_err = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(extract_video, r.src_path, str(out / r.frames_dir),
                      args.start_frame, args.num_frames, args.size, args.ext): r.video_id
            for r in idx.itertuples(index=False)
        }
        for k, f in enumerate(as_completed(futs), 1):
            s = f.result()
            s["video_id"] = futs[f]
            stats.append(s)
            if not s["ok"]:
                n_err += 1
                if n_err <= 5:
                    print(f"[extract] ERRORE {futs[f]}: {s.get('error')}")
                if n_err >= 20 and "scrittura" in str(s.get("error")):
                    for x in futs:
                        x.cancel()
                    raise SystemExit("[extract] Troppi errori di scrittura: interrotto. "
                                     "Controlla lo spazio libero sul disco o la cartella di output.")
            if k % 200 == 0 or k == len(futs):
                print(f"[extract] {k}/{len(futs)}")

    st = pd.DataFrame(stats).drop(columns=["src_path"])
    idx = idx.merge(st, on="video_id", how="left")

    bad = idx[~idx.ok.astype(bool)]
    if len(bad):
        print(f"[extract] ATTENZIONE: {len(bad)} video con problemi -> vedi videos_failed.csv")
        bad.to_csv(out / "videos_failed.csv", index=False)
    return idx


# ----------------------------------------------------------------------------
# 3. WINDOWS
# ----------------------------------------------------------------------------
def fix_identity(idx: pd.DataFrame) -> pd.DataFrame:
    """I metadati dei fake hanno yt_id troncati al primo '_' (gli ID YouTube possono contenere '_').
    Ogni video eredita quindi lo yt_id del real corrispondente, letto tramite clip_id."""
    real_yt = idx[idx.label == 0].set_index("clip_id")["yt_id"]
    idx = idx.copy()
    old = idx.yt_id.copy()
    idx["yt_id"] = idx.clip_id.map(real_yt)
    changed = (old != idx.yt_id).sum()
    if changed:
        print(f"[identity] corretti {changed} yt_id dei fake (presi dal real corrispondente)")
    assert idx.yt_id.notna().all(), "esistono video senza real corrispondente"
    assert idx.groupby("clip_id").yt_id.nunique().max() == 1
    return idx


def make_windows(idx: pd.DataFrame, args) -> pd.DataFrame:
    idx = fix_identity(idx)
    first, last = args.start_frame, args.start_frame + args.num_frames  # [first, last)
    starts = list(range(first, last - args.win_len + 1, args.win_stride))
    print(f"[windows] lunghezza {args.win_len}, stride {args.win_stride}: "
          f"{len(starts)} finestre per video, frame iniziali {starts}")

    ok = idx
    if "ok" in idx:
        # design paired: se anche un solo video di un clip (il real o uno dei fake) e' incompleto,
        # si esclude l'intero clip, cosi' ogni identita' rimasta ha 1 real + tutte le azioni.
        bad_clips = set(idx.loc[~idx.ok.astype(bool), "clip_id"])
        if bad_clips:
            print(f"[windows] esclusi {len(bad_clips)} clip incompleti (real + tutti i fake): "
                  f"restano {idx.clip_id.nunique() - len(bad_clips)} identita'")
            pd.Series(sorted(bad_clips), name="clip_id").to_csv(Path(args.out) / "excluded_clips.csv", index=False)
        ok = idx[~idx.clip_id.isin(bad_clips)]
    rows = []
    for r in ok.itertuples(index=False):
        for w, s in enumerate(starts):
            rows.append({
                "window_id": f"{r.video_id}__w{w:02d}",
                "video_id": r.video_id, "clip_id": r.clip_id, "yt_id": r.yt_id,
                "label": r.label, "action": r.action, "category": r.category,
                "frames_dir": r.frames_dir, "win_idx": w,
                "frame_start": s, "frame_end": s + args.win_len - 1,  # inclusivo
            })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# 4. SPLITS / PROTOCOLLI
# ----------------------------------------------------------------------------
def identity_split(ids, seed, ratios):
    ids = sorted(ids)
    random.Random(seed).shuffle(ids)
    n = len(ids)
    n_tr, n_va = round(n * ratios[0]), round(n * ratios[1])
    m = {i: "train" for i in ids[:n_tr]}
    m.update({i: "val" for i in ids[n_tr:n_tr + n_va]})
    m.update({i: "test" for i in ids[n_tr + n_va:]})
    return m


def write_split(df, path: Path, name: str, summary: dict):
    path.mkdir(parents=True, exist_ok=True)
    df.to_csv(path / f"{name}.csv", index=False)
    key = f"{path.name}/{name}"
    summary[key] = {
        "windows": len(df),
        "videos": int(df.video_id.nunique()),
        "identities": int(df.yt_id.nunique()),
        "real_windows": int((df.label == 0).sum()),
        "fake_windows": int((df.label == 1).sum()),
        "actions": sorted(df.loc[df.label == 1, "action"].unique().tolist()),
    }


def make_protocols(win: pd.DataFrame, out: Path, args) -> dict:
    split_of = identity_split(win.yt_id.unique(), args.seed, args.ratios)
    win = win.copy()
    win["id_split"] = win.yt_id.map(split_of)
    sp_dir = out / "splits"
    summary = {}

    real = win.label == 0
    tr, va, te = (win.id_split == s for s in ("train", "val", "test"))

    # --- P1 Identity Shift: tutte le azioni, identita' di test disgiunte ---
    p1 = sp_dir / "P1_identity_shift"
    write_split(win[tr], p1, "train", summary)
    write_split(win[va], p1, "val", summary)
    write_split(win[te], p1, "test", summary)

    # --- P2 / P3: leave-one-action-out e leave-one-category-out -----------
    # train e val sono comuni a P2 e P3 (si allena una volta per fold).
    # I real di test sono SEMPRE quelli delle identita' di test (mai visti),
    # quindi sono identici in P1, P2, P3: cambia solo il lato fake.
    #   P2: fake dell'azione esclusa generati da identita' di TRAIN (identita' vista)
    #   P3: fake dell'azione esclusa generati da identita' di TEST  (identita' non vista)
    folds = [("action", a) for a in sorted(win.loc[~real, "action"].unique())]
    folds += [("category", c) for c in sorted(win.loc[~real, "category"].unique())]

    for kind, held in folds:
        held_mask = (win[kind] == held) & ~real
        seen = real | ~held_mask
        d = sp_dir / f"LO{kind[0].upper()}O_{held}"   # LOAO_<azione> / LOCO_<categoria>
        write_split(win[tr & seen], d, "train", summary)
        write_split(win[va & seen], d, "val", summary)
        write_split(win[(te & real) | (tr & held_mask)], d, "test_P2_action_shift", summary)
        write_split(win[te & (real | held_mask)], d, "test_P3_identity_action_shift", summary)

    # controllo anti-leakage: in P1 e P3 nessuna identita' di test deve comparire in train
    for d in sp_dir.iterdir():
        if not d.is_dir():
            continue
        tr_ids = set(pd.read_csv(d / "train.csv", usecols=["yt_id"]).yt_id)
        for t in d.glob("test*.csv"):
            if "P2" in t.name:
                continue
            leak = tr_ids & set(pd.read_csv(t, usecols=["yt_id"]).yt_id)
            assert not leak, f"LEAKAGE in {d.name}/{t.name}: {len(leak)} identita' condivise con train"
    n_ids = pd.Series(split_of).value_counts()
    print(f"[splits] identita': {len(split_of)} totali -> "
          f"train {n_ids.get('train', 0)}, val {n_ids.get('val', 0)}, test {n_ids.get('test', 0)}; "
          f"controllo leakage OK")

    pd.Series(split_of, name="id_split").rename_axis("yt_id").to_csv(sp_dir / "identity_split.csv")
    return summary


# ----------------------------------------------------------------------------
# 5. QA
# ----------------------------------------------------------------------------
def qa_report(idx: pd.DataFrame, out: Path):
    if "codec" not in idx:
        return
    g = idx.assign(group=np.where(idx.label == 0, "real", "fake"))
    rep = g.groupby("group").agg(
        n=("video_id", "count"),
        codecs=("codec", lambda s: dict(s.value_counts())),
        fps=("fps", lambda s: dict(s.value_counts())),
        frame_count_min=("frame_count", "min"),
        frame_count_median=("frame_count", "median"),
        frame_count_max=("frame_count", "max"),
        width_median=("width", "median"),
        bitrate_kbps_median=("bitrate_kbps", "median"),
        bitrate_kbps_p10=("bitrate_kbps", lambda s: s.quantile(.1)),
        bitrate_kbps_p90=("bitrate_kbps", lambda s: s.quantile(.9)),
    )
    rep.to_csv(out / "qa_report.csv")
    print("\n[QA] confronto real vs fake (file originali, prima del preprocessing):")
    print(rep.T.to_string())

    # bitrate per coppia real/fake: utile per individuare differenze di compressione
    r = idx[idx.label == 0].set_index("clip_id")["bitrate_kbps"]
    f = idx[idx.label == 1][["clip_id", "action", "bitrate_kbps"]].copy()
    f["real_bitrate_kbps"] = f.clip_id.map(r)
    f["ratio_fake_over_real"] = f.bitrate_kbps / f.real_bitrate_kbps
    f.to_csv(out / "qa_bitrate_pairs.csv", index=False)
    print("\n[QA] rapporto bitrate fake/real per azione (mediana):")
    print(f.groupby("action").ratio_fake_over_real.median().round(2).to_string())


# ----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=".", help="cartella che contiene final_dataset/ e generated/")
    p.add_argument("--out", default=r"D:\benchmark", help="cartella di output (default D:\benchmark)")
    p.add_argument("--start-frame", type=int, default=1, help="primo frame usato (default 1: salta il first frame)")
    p.add_argument("--num-frames", type=int, default=72, help="numero di frame per video (default 72 -> frame 1..72)")
    p.add_argument("--size", type=int, default=256, help="lato del frame dopo il resize")
    p.add_argument("--ext", choices=["png", "jpg"], default="jpg")
    p.add_argument("--win-len", type=int, default=16, help="lunghezza sliding window")
    p.add_argument("--win-stride", type=int, default=8, help="stride sliding window")
    p.add_argument("--ratios", type=float, nargs=3, default=[0.7, 0.1, 0.2], help="train val test per identita'")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--force", action="store_true", help="ignora il controllo dello spazio su disco")
    p.add_argument("--skip-extract", action="store_true",
                   help="riusa index_videos.csv e i frame gia' estratti; rigenera solo finestre e split")
    args = p.parse_args()

    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.skip_extract:
        idx = pd.read_csv(out / "index_videos.csv")
        print(f"[index] caricato index_videos.csv ({len(idx)} video)")
    else:
        idx = build_index(root)
        idx = run_extract(idx, out, args)
        idx.to_csv(out / "index_videos.csv", index=False)
        qa_report(idx, out)

    win = make_windows(idx, args)
    win.to_csv(out / "index_windows.csv", index=False)
    summary = make_protocols(win, out, args)

    config = {k: v for k, v in vars(args).items()}
    with open(out / "splits" / "summary.json", "w") as fh:
        json.dump({"config": config, "splits": summary}, fh, indent=2)

    print("\n[splits] riepilogo:")
    for k, v in summary.items():
        print(f"  {k:55s} finestre={v['windows']:6d}  real={v['real_windows']:5d}  "
              f"fake={v['fake_windows']:6d}  id={v['identities']}")
    print(f"\nFatto. Output in: {out.resolve()}")


if __name__ == "__main__":
    main()