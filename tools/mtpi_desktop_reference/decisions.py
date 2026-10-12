import json

from native import run, u

initial = u.context_save()
rows = []
for target in (0.1, 0.25, 0.3, 0.5):
    for ncurrent in (1, 3, 6, 12):
        for ycurrent in range(ncurrent + 1):
            for d in range(3):
                for neighbor in (0, 1, 3, 6):
                    for tt in (0, 1):
                        n = [3] * 3
                        y = [0] * 3
                        n[d] = ncurrent
                        y[d] = ycurrent
                        if d < 2:
                            n[d + 1] = neighbor
                            y[d + 1] = neighbor
                        u.context_restore(initial)
                        rows.append(run(n, y, d, target, tt=tt))
print(json.dumps(rows, indent=2))
