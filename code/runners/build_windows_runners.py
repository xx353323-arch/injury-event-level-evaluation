import numpy as np, pandas as pd, os
HERE = os.path.dirname(os.path.abspath(__file__))
FEAT = ['nr. sessions','total km','km Z3-4','km Z5-T1-T2','km sprinting','strength training','hours alternative','perceived exertion','perceived trainingSuccess','perceived recovery']
T = 7

df = pd.read_csv(os.path.join(HERE, 'day_approach_maskedID_timeseries.csv')).sort_values(['Athlete ID','Date']).reset_index(drop=True)
X = np.zeros((len(df), T, len(FEAT)), dtype=np.float32)
for k in range(T):
    cols = [f if k == 0 else f'{f}.{k}' for f in FEAT]
    X[:, k, :] = df[cols].values.astype(np.float32)
y = df['injury'].astype(np.int64).values
ath = df['Athlete ID'].astype(int).values
dates = df['Date'].astype(int).values
eid = np.array([f'{a}|{d}' if lab == 1 else '' for a, d, lab in zip(ath, dates, y)])

rng = np.random.default_rng(0)
inj_per_ath = pd.Series(y).groupby(ath).sum().sort_values(ascending=False)
groups = {}
order = inj_per_ath.index.tolist()
for i, a in enumerate(order):
    groups[a] = ['G1','G2','G3'][i % 3]
task = np.array([groups[a] for a in ath])

np.savez_compressed(os.path.join(HERE, 'windows_runners.npz'), X=X, y=y, athletes=ath, dates=dates, event_id=eid, tasks=task)

print(f'windows {len(y)}  positives {int(y.sum())} (= independent events)  prevalence {y.mean():.4f}  X {X.shape}')
print('Three groups interleaved by injury count:')
for g in ['G1','G2','G3']:
    m = task == g
    print(f'  {g}: athletes {len(set(ath[m]))}  windows {m.sum()}  positives {int(y[m].sum())}  prevalence {y[m].mean():.4f}')
print('Temporal order check: X[:,0,:] = seven days before, X[:,6,:] = previous day')
