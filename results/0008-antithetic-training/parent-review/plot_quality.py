"""Plot all published 0008 case gains; no model or frozen scorer invocation."""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser()
p.add_argument('--scores',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
data=json.loads(a.scores.read_text())
assert data['status']=='stop_this_candidate' and data['case_count']==64
assert hashlib.sha256(a.scores.read_bytes()).hexdigest()=='810cfd9384c475dc0df3d048bcd98a5bd33f96993e70e15d6ef69d68e40943e7'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(10.4,4.7),sharey=True)
for ax,key,title in zip(axes,('updates500','wall5s'),('500 optimizer updates','Approximately 5 seconds of training')):
    endpoint=data['endpoints'][key]
    for i,g in enumerate(('d1','d2','d3','d4')):
        pairs=[x for x in endpoint['pairs'] if x['group']==g]
        assert [x['seed'] for x in pairs]==[0,1]
        for pair,offset,marker in zip(pairs,(-.1,.1),('o','s')):
            ax.scatter(i+offset,pair['gain_db'],s=44,marker=marker,color='#9e3f2d',edgecolors='white',linewidths=.6,zorder=4)
        ax.plot([i-.22,i+.22],[endpoint['group_mean_gain_db'][g]]*2,color='#263744',lw=2.5,zorder=3)
    ax.axhline(0,color='#76808a',lw=.8)
    ax.axhline(.5,color='#336c63',ls='--',lw=1.1)
    ax.set_xticks(range(4),['d1','d2','d3','d4'])
    ax.set_xlim(-.5,3.5)
    ax.set_title(title,fontsize=11,loc='left',pad=12)
    ax.text(.02,.97,f"Overall mean: {endpoint['mean_gain_db']:+.4f} dB",transform=ax.transAxes,va='top',fontsize=10,color='#29333d')
    ax.grid(axis='y',color='#e4e7eb',lw=.6)
    ax.set_axisbelow(True)
axes[0].set_ylabel('PSNR difference from per-case best control (dB)')
axes[0].set_ylim(-1.85,1.8)
axes[1].text(3.37,.54,'Required mean: +0.5 dB',ha='right',color='#336c63',fontsize=9)
fig.suptitle('0008 · Mean pairing did not meet the denoising-quality gate',x=.075,ha='left',fontsize=14,y=.98)
fig.text(.075,.12,'Circle / square: the two fixed noise seeds. Dark bar: image-token mean. Every case is shown.',fontsize=9)
fig.text(.075,.074,'Four reused development images; synthetic known covariance. Endpoints share trajectories; no population interval.',fontsize=9,color='#525b65')
fig.subplots_adjust(left=.075,right=.985,top=.81,bottom=.23,wspace=.13)
fig.savefig(a.output,dpi=180,facecolor='white')
plt.close(fig)
print(json.dumps({'score_sha256':hashlib.sha256(a.scores.read_bytes()).hexdigest(),'output_sha256':hashlib.sha256(a.output.read_bytes()).hexdigest(),'points':16,'model_calls':0}))
