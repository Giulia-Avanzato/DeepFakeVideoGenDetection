# DeepFakeVideoGenDetection

Context:

Recent advances in generative Artificial Intelligence have
led to the rapid development of Image-to-Video (I2V) dif-
fusion models, which are emerging as a new paradigm for
facial deepfake generation. Unlike conventional deepfake
techniques, such as face swapping, talking-head synthesis, or
audio-driven facial animation, I2V diffusion models generate
realistic facial videos from a single authentic image while
producing temporally coherent facial expressions and head
movements. This capability significantly lowers the effort
required to create convincing manipulated videos and intro-
duces a new and increasingly realistic threat model for facial
forgery.
Beyond improving visual realism, modern I2V diffusion
models provide explicit semantic control over facial expres-
sions and head movements through action prompts, enabling
the generation of a broad range of action-conditioned facial
forgeries. As these models continue to improve in quality and
controllability, understanding the robustness of current state-
of-the-art deepfake detectors against this emerging generation
paradigm has become increasingly important.


Current version of the code #########################

Action-Conditioned I2V Deepfake Benchmark — Data Preparation

prepare_benchmark.py builds a paired deepfake detection benchmark from authentic CelebV-HQ clips (real) and their WAN 2.2 image-to-video counterparts (fake), each conditioned on a different facial or head action. It extracts frames, builds sliding windows, and generates identity-disjoint splits for three evaluation protocols: Identity Shift, Action Shift and Identity+Action Shift.

Dataset at a glance
	
Real videos	405 CelebV-HQ clips (one identity per clip)
Fake videos	9 actions × 405 = 3,645 WAN 2.2 I2V videos
Action categories	expressions (angry, laugh, surprise) · head_movements (head_turn, look_aside, nod) · mouth_movements (kiss, smile, talk)
Pairing	every fake is generated from the first frame of its real clip (same clip_id)
After filtering	378 identities × (1 real + 9 fakes) = 3,780 videos
Windows	8 per video (length 16, stride 8) = 30,240 windows
Requirements
python -m venv D:\venvs\deepfake
D:\venvs\deepfake\Scripts\Activate.ps1
pip install opencv-python pandas numpy scikit-learn

Python ≥ 3.9. No GPU is needed for this step.

Expected input structure

Run the script from the folder that contains final_dataset/ and generated/ (or pass it with --root):

<root>/
├── final_dataset/final_dataset/
│   ├── metadata_final.csv            # real clips: clip_id, yt_id, processed_path, fps, ...
│   ├── processed_videos/<clip_id>.mp4
│   └── first_frames/<clip_id>_first_frame.png
└── generated/generated/
    └── <category>/<action>/
        ├── metadata_<action>.csv     # fake videos: file_name, clip_id, yt_id, ...
        └── videos/NNN_<action>_<clip_id>.mp4
Usage

Full run (index → frame extraction → windows → splits → QA):

python prepare_benchmark.py

By default, output goes to D:\benchmark and frames are saved as quality-95 JPEG.

To regenerate only windows and splits, without re-extracting frames:

python prepare_benchmark.py --skip-extract
python prepare_benchmark.py --skip-extract --win-len 32 --win-stride 16
python prepare_benchmark.py --skip-extract --seed 1 --out D:\benchmark_seed1

Note that --out must point to a folder that already contains extracted frames when --skip-extract is used.

Options
Option	Default	Description
--root	.	folder containing final_dataset/ and generated/
--out	D:\benchmark	output folder
--start-frame	1	first frame used (frame 0 is skipped, see below)
--num-frames	72	frames kept per video (frames 1–72)
--size	256	frame side after resizing (square)
--ext	jpg	jpg (quality 95, ~3 GB) or png (lossless, ~31 GB)
--win-len	16	sliding-window length (frames)
--win-stride	8	sliding-window stride (frames)
--ratios	0.7 0.1 0.2	train / val / test fractions over identities
--seed	42	random seed for the identity split
--workers	#CPU − 1	parallel processes for frame extraction
--skip-extract	off	reuse index_videos.csv and existing frames
--force	off	skip the free-disk-space check
Pipeline
1. Index (build_index)
Reads metadata_final.csv (real) and every metadata_<action>.csv (fake).
Action and category names are taken from the folder names (e.g. head_movements/head_turn), not from file names, which use inconsistent spellings (headturn).
Assigns a unique video_id (real__<clip_id> or <action>__<clip_id>) and a label (0 = real, 1 = fake).
Reports missing files and drops fakes without a matching real clip.
2. Frame extraction (run_extract, extract_video)
Estimates the required disk space and aborts if it is insufficient.
For each video, in parallel:
reads codec, fps, frame count, resolution and bitrate (file size / duration) for the QA report;
keeps frames 1–72. Frame 0 is skipped because in each fake it is identical to the real source frame given to WAN 2.2. Real videos use the same indices, so each real/fake pair covers the same temporal span starting from the same source frame;
resizes every frame to 256×256 with cv2.INTER_AREA;
writes frames/<action|real>/<clip_id>/001.jpg … 072.jpg.
Real and fake videos go through exactly the same pipeline (decode → resize → encode), so no real/fake difference is introduced by preprocessing.
Videos with fewer than 73 frames are flagged as incomplete. Write failures are reported explicitly, and the run aborts after repeated failures.
3. Sliding windows (fix_identity, make_windows)
Identity fix: in the fake metadata, yt_id was truncated at the first underscore, but YouTube IDs may contain _ (e.g. --uyzf7X_0c). Every video therefore inherits the yt_id of its real counterpart via clip_id. This affected 747 fake videos (83 clips); without the fix, real and fake videos of the same identity could fall into different splits.
Paired filtering: if any video of a clip (the real one or any of its 9 fakes) is incomplete, the whole clip is removed. This removed 27 clips, mostly real videos shorter than 73 frames, leaving 378 identities. The removed clips are listed in excluded_clips.csv.
Windows: with length 16 and stride 8 over frames 1–72, each video yields 8 overlapping windows starting at frames 1, 9, 17, 25, 33, 41, 49 and 57.
Windows are stored as rows of an index (frames_dir, frame_start, frame_end); no frame is duplicated on disk.
4. Splits and protocols (make_protocols)
Identities (yt_id) are shuffled with a fixed seed and split 70 / 10 / 20 % into train (265) / val (38) / test (75). All windows of an identity (real + 9 fakes) stay in the same subset.
Protocol	Train	Test
P1 — Identity Shift	train identities; real + all 9 actions	test identities; real + all 9 actions
P2 — Action Shift	train identities; real + seen actions	real videos of test identities + fakes of the held-out action generated from train identities
P3 — Identity+Action Shift	same as P2	test identities; real + fakes of the held-out action

P2 and P3 are generated for:

LOAO (leave-one-action-out): 9 folds, one per action;
LOCO (leave-one-category-out): 3 folds, one per category.

Design notes:

P2 and P3 share the same train and validation sets within a fold: train once, evaluate on both test sets.
The real test set is identical across all protocols (the 75 test identities, never seen in training), so differences between protocols come only from the fake side.
In P2, real test videos come from test identities to avoid evaluating on real videos memorized during training. Each real video is shared by all actions of its identity, so using train-identity reals would leak.
An anti-leakage check asserts that no test identity appears in the training set of P1 and P3; the script stops otherwise.
5. Quality report (qa_report)

Compares the original real and fake files in terms of codec, fps, frame count, resolution and bitrate.

Output
D:\benchmark\
├── frames\<action|real>\<clip_id>\001.jpg … 072.jpg
├── index_videos.csv          # one row per video (metadata + extraction stats)
├── index_windows.csv         # one row per window
├── excluded_clips.csv        # clips removed by paired filtering
├── videos_failed.csv         # videos with extraction problems (if any)
├── qa_report.csv             # real vs fake comparison (codec, fps, frames, bitrate)
├── qa_bitrate_pairs.csv      # per-pair bitrate and fake/real ratio
└── splits\
    ├── identity_split.csv    # yt_id -> train / val / test
    ├── summary.json          # configuration + counts for every split
    ├── P1_identity_shift\{train,val,test}.csv
    ├── LOAO_<action>\{train,val,test_P2_action_shift,test_P3_identity_action_shift}.csv
    └── LOCO_<category>\{train,val,test_P2_action_shift,test_P3_identity_action_shift}.csv
Window CSV columns (index_windows.csv and every split file)
Column	Description
window_id	unique window id (<video_id>__wNN)
video_id	unique video id; use it to aggregate window scores into video scores
clip_id	CelebV-HQ clip id, shared by a real video and its 9 fakes
yt_id	identity (YouTube video id)
label	0 = real, 1 = fake
action / category	prompted action and its category (real for real videos)
frames_dir	frame folder, relative to the output folder
win_idx	window index within the video (0–7)
frame_start, frame_end	first and last frame of the window (inclusive)
id_split	train / val / test (split files only)

To load a window, read frame_start … frame_end from <out>/<frames_dir>/NNN.jpg.

Split sizes (default settings, seed 42)
Split	Windows	Real	Fake	Identities
P1 train / val / test	21,200 / 3,040 / 6,000	2,120 / 304 / 600	19,080 / 2,736 / 5,400	265 / 38 / 75
LOAO train (each fold)	19,080	2,120	16,960	265
LOAO test P2 / P3	2,720 / 1,200	600 / 600	2,120 / 600	340 / 75
LOCO train (each fold)	14,840	2,120	12,720	265
LOCO test P2 / P3	6,960 / 2,400	600 / 600	6,360 / 1,800	340 / 75
Quality assurance findings
	Real	Fake
Codec	H.264	H.264
fps	23.976–60 (same per pair)	same as the paired real
Resolution (median)	790 px	790 px (same per pair)
Frames	61–1199 (median 107)	73
Bitrate (median)	1,192 kbps	790 kbps
Duration differences are removed by using frames 1–72 for both real and fake videos.
Compression: fakes are encoded at a lower bitrate than real videos, and the gap depends on the action (median 638 kbps for look_aside, 1,264 kbps for laugh). Bitrate alone separates real from fake with AUC = 0.66. Resizing to 256 px and re-encoding all frames with the same JPEG settings attenuates this cue. A compression-harmonized control experiment (re-encoding real and fake segments with identical H.264 settings) is planned to rule out compression as a confounder in the action-wise analysis.
Notes and limitations
CelebV-HQ provides no explicit identity labels. Identities are approximated by yt_id, and different YouTube videos of the same person cannot be detected.
The class ratio is 1 real : 9 fake in training. Use class weighting or balanced sampling, and report threshold-free metrics (AUC, AP).
With 75 test identities, per-action results rely on 75 fake videos each. Report bootstrap confidence intervals over identities, or repeat with several seeds (--seed).
