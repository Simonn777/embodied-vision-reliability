"""Plot all published per-case B variance ratios; no private arrays or model calls."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root=Path(__file__).resolve().parents[1]
p=root/'study/scores.json'
assert hashlib.sha256(p.read_bytes()).hexdigest()=='ff4542f059ae02cce36aeacfda2d5346075e524d7863e8666b5346756c8306a9'
s=json.loads(p.read_text())
rows=[x for x in s['comparisons'] if x['regime']=='B_total']
fig,ax=plt.subplots(figsize=(8,4.2))
fig.subplots_adjust(left=.20,right=.97,bottom=.30,top=.84)
for i,row in enumerate(rows):
    raw=np.asarray(row['per_case_variance_ratio'])
    ax.scatter(raw, i+np.linspace(-.13,.13,8), s=26,color='#b6c2cc',edgecolors='white',linewidth=.4,zorder=3)
    v=row['variance_ratio_geomean'];upper=row['bootstrap_upper']
    ax.plot([v,upper],[i,i],color='#176a70',linewidth=2.2,zorder=4)
    ax.scatter([v],[i],marker='D',s=42,color='#176a70',zorder=5)
    ax.text(1.2,i,f'{v:.3f}',va='center',fontsize=10,color='#176a70')
ax.axvline(1,color='#a25e40',linewidth=1,linestyle='--')
ax.axvline(.9,color='#87939c',linewidth=.8,linestyle=':')
ax.set_xscale('log');ax.set_xlim(.12,1.7)
ax.set_xticks([.125,.25,.5,.9,1]);ax.set_xticklabels(['0.125','0.25','0.5','0.9','1.0'])
ax.set_yticks(range(3));ax.set_yticklabels(['vs independent','vs random pairing','vs same-patch'])
ax.invert_yaxis();ax.set_ylim(2.5,-.6)
ax.set_xlabel('Mean-pair gradient variance / control variance  (lower is better)',labelpad=10)
ax.set_title('Lower total gradient variance at the saved checkpoints',loc='left',pad=18,fontweight='bold')
ax.spines[['top','right','left']].set_visible(False)
ax.tick_params(axis='y',length=0);ax.grid(axis='x',alpha=.13)
fig.text(.02,.02,'Eight fixed development checkpoints · 64 repeats · gray: every case · diamond: geometric mean\nLine: one-sided bootstrap upper bound (98.33% quantile); conditional Monte Carlo uncertainty only.\nDotted: 0.90 screen threshold; dashed: equal variance. No image-quality improvement measured.',fontsize=8,color='#48535b')
fig.savefig(root/'gradient-variance.png',dpi=180,bbox_inches='tight',facecolor='white')
plt.close(fig)
