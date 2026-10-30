"""Lance tout le pipeline dans l'ordre :  python run_all.py   (depuis le dossier src)"""
import subprocess, sys, time
from pathlib import Path
here = Path(__file__).resolve().parent
steps = ["01_extract.py", "02_eda.py", "03_events.py", "04_baseline.py", "05_gnn.py", "06_compare.py"]
if "--quick" in sys.argv:
    steps[4] = "05_gnn.py --quick"
for s in steps:
    t0 = time.time(); print("\n" + "=" * 70 + "\n>>> " + s + "\n" + "=" * 70, flush=True)
    parts = s.split(); r = subprocess.run([sys.executable, str(here / parts[0])] + parts[1:], cwd=str(here))
    if r.returncode != 0:
        sys.exit("Échec à l'étape %s" % s)
    print("--- %s terminé en %.0fs" % (s, time.time() - t0), flush=True)
print("\nTERMINÉ. Résultats : results/tables et results/figures")
