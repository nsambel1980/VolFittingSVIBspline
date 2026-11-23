"""Day-to-day stability of the modes.

1. Cosines between the leading modes of the first day and their best match on each later day
   (in the P1 metric, so the comparison does not depend on the coefficient scaling).
2. Refit each later day with the FIRST day's modes held fixed; the free-energy cost of not recomputing them.
Writes results/stability/stability.json and stability.md.
"""
import numpy as np
from _common import chains, day_of, out_dir, dump, markdown_table
from volsurf import Config, run_day
from volsurf.dupire import LocalVarSurface
from volsurf.fit import baseline

cfg = Config()
files = chains()
if len(files) < 2:
    raise SystemExit("needs at least two chains")
ref = run_day(files[0], cfg, relin=False)
V0 = ref["modes"]["V"][:, :cfg.n_modes]
P1 = LocalVarSurface(ref["mk"], baseline(ref["mk"], ref["base"]["q"], cfg), cfg, a=np.zeros(V0.shape[0])).P1
out, rows = {}, []
for path in files[1:]:
    day = day_of(path); print(f"{day}:")
    own = run_day(path, cfg, relin=False)
    V1 = own["modes"]["V"][:, :cfg.n_modes]
    G = np.abs(V0.T @ P1 @ V1)
    cos = [float(G[i].max()) for i in range(8)]
    fixed = run_day(path, cfg, modes=dict(V=V0), relin=False)
    dF = fixed["fit"]["F"] - own["fit"]["F"]
    out[day] = dict(cosines_top8=cos, F_own_modes=own["fit"]["F"], F_fixed_modes=fixed["fit"]["F"], dF=dF)
    rows.append([day, " ".join(f"{c:.2f}" for c in cos[:4]), f"{own['fit']['F']:.1f}",
                 f"{fixed['fit']['F']:.1f}", f"{dF:+.1f}"])
d = out_dir("stability")
dump(out, d / "stability.json")
(d / "stability.md").write_text(f"Reference modes: {day_of(files[0])}\n\n" + markdown_table(
    rows, ["day", "cosines, modes 1-4", "F own modes", f"F with {day_of(files[0])} modes", "cost"]))
print((d / "stability.md").read_text())
