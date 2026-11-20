"""Run the full pipeline in order.

Usage:
    python run_all.py                 # all steps
    python run_all.py --from 07       # resume from a step
    python run_all.py --only 09 10    # selected steps

Set DATA_DIR (raw datasets) and WORK_DIR (outputs) as environment variables;
on Kaggle the defaults (/kaggle/input, /kaggle/working) are used.
"""
import argparse, glob, os, runpy, sys, time

os.environ.setdefault("MPLBACKEND", "Agg")
HERE = os.path.dirname(os.path.abspath(__file__))
STEPS = sorted(glob.glob(os.path.join(HERE, "src", "[0-9][0-9]*_*.py")))

ap = argparse.ArgumentParser()
ap.add_argument("--from", dest="start", default=None, help="first step to run, e.g. 07")
ap.add_argument("--only", nargs="*", default=None, help="run only these steps, e.g. 09 10")
args = ap.parse_args()

def step_id(path):
    return os.path.basename(path).split("_")[0]

todo = [s for s in STEPS
        if (args.only is None or step_id(s) in args.only)
        and (args.start is None or step_id(s) >= args.start)]
if not os.path.isdir("/kaggle/input"):          # local run: keep everything inside the repository
    os.environ.setdefault("WORK_DIR", os.path.join(HERE, "outputs"))
    os.environ.setdefault("DATA_DIR", os.path.join(HERE, "data", "raw"))
os.environ.setdefault("WORK_DIR", "/kaggle/working"); os.environ.setdefault("DATA_DIR", "/kaggle/input")
os.makedirs(os.environ["WORK_DIR"], exist_ok=True)
print(f"DATA_DIR = {os.environ['DATA_DIR']}\nWORK_DIR = {os.environ['WORK_DIR']}\n")

for s in todo:
    t0 = time.time()
    print("=" * 72 + f"\n{os.path.basename(s)}\n" + "=" * 72, flush=True)
    try:
        runpy.run_path(s, run_name="__main__")
    except Exception as e:
        sys.exit(f"\nStep {os.path.basename(s)} failed: {e!r}")
    print(f"\n[{os.path.basename(s)} finished in {(time.time() - t0) / 60:.1f} min]\n", flush=True)
