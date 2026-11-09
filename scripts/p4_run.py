"""
p4_run.py  -  run a group of Phase-4 experiments one after another without a notebook (e.g. overnight).

    (.venv) python scripts/p4_run.py kd          # one group
    (.venv) python scripts/p4_run.py kd adapters # several groups, in this order

Runs exactly the same code as the notebooks (src/cjpe_experiment.py) with the same configs, so the results land
in the same run folders (checkpoints/phase4/<run>/). Opening the notebook afterwards just LOADS everything
("[cache] ..."), and its outputs show the results.
If one experiment fails, the error is printed and the next one still runs. Re-running continues where it stopped
(training resumes from the last checkpoint, finished steps are skipped).
Everything printed also goes to experiments/phase4/run_log.txt.
"""

import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
while not (ROOT / ".gitignore").exists():
    if ROOT.parent == ROOT:
        raise FileNotFoundError("Project root with .gitignore not found")
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

from cjpe_experiment import Experiment, compare   # noqa: E402

TEACHER = "A2_full_causal_last1536_attn"      # experiment 1 = KD teacher
GROUPS = {
    # group: [(notebook, config, also save teacher outputs for KD?), ...]
    "night1": [("P4_02_A2_full", {"phase1": "A2", "method": "full"}, True),
               ("P4_03_B_lorma", {"phase1": "B", "method": "lorma"}, False)],
    "kd": [("P4_04_D_kd", {"phase1": "D", "method": "full", "kd_teacher": TEACHER}, False),   # student + KD
           ("P4_04_D_kd", {"phase1": "D", "method": "full"}, False)],                         # control: no teacher
    "adapters": [("P4_05_A_adapters", {"phase1": "A", "method": "adapter"}, False)],
    "gradual": [("P4_06_A2_gradual", {"phase1": "A2", "method": "gradual"}, False)],
    "bitfit": [("P4_08_A2_bitfit", {"phase1": "A2", "method": "bitfit"}, False)],
}
names = sys.argv[1:] or ["kd"]
unknown = [n for n in names if n not in GROUPS]
if unknown:
    raise SystemExit(f"unknown group(s) {unknown}; available: {list(GROUPS)}")
EXPERIMENTS = [e for n in names for e in GROUPS[n]]

LOG = ROOT / "experiments" / "phase4" / "run_log.txt"
LOG.parent.mkdir(parents=True, exist_ok=True)


class Tee:
    """Print to the terminal AND append to the log file."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
            st.flush()

    def flush(self):
        for st in self.streams:
            st.flush()


log_file = open(LOG, "a", encoding="utf-8")
sys.stdout = Tee(sys.__stdout__, log_file)

for notebook, cfg, teacher in EXPERIMENTS:
    print(f"\n{'=' * 90}\n{time.strftime('%Y-%m-%d %H:%M')}  {notebook}  {cfg}\n{'=' * 90}")
    try:
        exp = Experiment(ROOT, cfg)
        exp.memory_check()
        exp.train()
        exp.report()
        exp.predict_test()
        if teacher:
            exp.teacher_outputs()
        exp.speed()
        exp.log()
        print(f"{time.strftime('%H:%M')}  {notebook} finished")
    except Exception:
        print(f"{time.strftime('%H:%M')}  {notebook} FAILED:")
        traceback.print_exc(file=sys.stdout)
    finally:
        try:
            import torch
            del exp
            torch.cuda.empty_cache()
        except Exception:
            pass

print("\n", compare(ROOT))
print(f"\nall done {time.strftime('%Y-%m-%d %H:%M')}")
