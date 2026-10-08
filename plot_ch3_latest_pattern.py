"""Plot either a fixed K=8 scene or an explicitly selected showcase case."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from beamnas.common import save_json, sha256


def select_showcase(angles, mask, run):
    results = {}
    for name in ['ours_optimized','MVDR_standard','DNNABF_standard']:
        with np.load(run/(name+'.npz')) as data:
            results[name] = {key:data[key].copy() for key in ['sinr_db','main_error_deg']}
    ours, mvdr, baseline = [results[key] for key in results]
    gap = mvdr['sinr_db']-ours['sinr_db']
    eligible = ((mask.sum(1)==9) & (abs(angles[:,0])<=60)
                & (ours['main_error_deg']<=.10000001)
                & (ours['main_error_deg']<=mvdr['main_error_deg']+1e-8)
                & (ours['main_error_deg']<=baseline['main_error_deg']+1e-8))
    indices = np.flatnonzero(eligible)
    if not len(indices):raise ValueError('No scene meets the declared showcase criteria')
    order = indices[np.argsort(gap[indices],kind='stable')]
    return int(order[0]), {
        'mode':'post_test_showcase_selected_at_user_request',
        'test_scenes':len(angles),'K8_scenes':int((mask.sum(1)==9).sum()),
        'eligible_scenes':len(indices),
        'eligibility':{'interferers':8,'abs_desired_AOA_max_deg':60,
                       'ours_main_error_max_deg':.1,
                       'ours_main_error_not_greater_than':['MVDR','DNNABF reconstruction']},
        'ranking':'Smallest MVDR-minus-ours SINR gap; tie broken by original test index',
        'top_candidates':[{'test_index':int(i),'sinr_gap_db':float(gap[i])} for i in order[:10]],
        'interpretation':'Selected favorable illustration, not representative sampling or independent evidence of overall performance; aggregate test results unchanged.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='output/pdf/ch3_latest_pattern')
    parser.add_argument('--selection',choices=['first','showcase'],default='first')
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    test = Path('data/ch3_grid_good_v11/test.npz')
    run = Path('runs/ch3_gpu_deployment_v1/full_test')
    report = json.loads((run/'report.json').read_text())
    assert report['complete'] and sha256(test) == report['test_sha256']
    with np.load(test) as data:
        if args.selection == 'showcase':
            index, selection = select_showcase(data['angles'],data['mask'],run)
        else:
            index = int(np.flatnonzero(data['mask'].sum(-1)-1 == 8)[0])
            selection = {'mode':'First test scene with K=8, no ranking by model results'}
        angles = data['angles'][index][data['mask'][index].astype(bool)].astype(float)
    desired, interference = angles[0], angles[1:]
    grid = np.linspace(-90,90,18001)
    steering = np.exp(1j*np.pi*np.sin(np.deg2rad(grid))[:,None]*np.arange(12))
    scene_steering = np.exp(1j*np.pi*np.sin(np.deg2rad(angles))[:,None]*np.arange(12))
    names = ['MVDR_standard','ours_optimized','DNNABF_standard']
    labels = ['MVDR','本方法','DNNABF（原始结构复现）']
    colors = ['#303030','#0072B2','#D55E00']
    styles = ['-','--','-.']
    curves, metrics, hashes = {}, {}, {}
    for name in names:
        p = run/(name+'.npz')
        hashes[str(p)] = sha256(p)
        with np.load(p) as data:
            w = data['weights'][index]
            power = abs(w.conj()@steering.T)**2
            pattern = 10*np.log10(np.maximum(power/power.max(),1e-30))
            response = abs(w.conj()@scene_steering.T)**2
            sinr = 10*np.log10(10*response[0]/(1000*response[1:].sum()+abs(w).dot(abs(w))))
            assert abs(sinr-float(data['sinr_db'][index])) < 1e-8
            metrics[name] = {'sinr_db':float(sinr),'nearest_peak_deg':float(data['peak_deg'][index]),
                             'main_error_deg':float(data['main_error_deg'][index]),
                             'weight_real':w.real.tolist(),'weight_imag':w.imag.tolist(),
                             'global_peak_on_plot_grid_deg':float(grid[power.argmax()]),
                             'gain_at_desired_db_relative_own_peak':float(10*np.log10(response[0]/power.max())),
                             'gain_at_interference_db_relative_own_peak':(10*np.log10(response[1:]/power.max())).tolist()}
            curves[name] = pattern
    plt.rcParams.update({'font.family':'Noto Sans CJK JP','font.size':10,
                         'axes.unicode_minus':False,'pdf.fonttype':3,'ps.fonttype':3,
                         'axes.spines.top':False,'axes.spines.right':False})
    fig = plt.figure(figsize=(12,7.2), facecolor='white')
    gs = fig.add_gridspec(1,2,left=.07,right=.98,bottom=.32,top=.755,wspace=.24,width_ratios=[2.15,1])
    ax, zoom = fig.add_subplot(gs[0,0]),fig.add_subplot(gs[0,1])
    handles = []
    for name,label,color,style in zip(names,labels,colors,styles):
        line, = ax.plot(grid,curves[name],label=label,color=color,linestyle=style,
                        linewidth=1.6 if name != 'MVDR_standard' else 1.2)
        handles.append(line)
        zoom.plot(grid,curves[name],color=color,linestyle=style,
                  linewidth=1.7 if name != 'MVDR_standard' else 1.3)
        peak = metrics[name]['nearest_peak_deg']
        peak_index = int(np.argmin(abs(grid-peak)))
        zoom.scatter([peak],[curves[name][peak_index]],s=22,color=color,zorder=5)
    for theta in interference:ax.axvline(theta,color='#A5A5A5',linestyle=':',linewidth=.75,zorder=0)
    for panel in [ax,zoom]:
        panel.axvline(desired,color='#009E73',linestyle=(0,(4,3)),linewidth=1.0,zorder=0)
        panel.grid(True,alpha=.18,linewidth=.6)
        panel.set_xlabel('扫描角度 (°)')
    ax.set(xlim=(-90,90),ylim=(-100,2),xticks=np.arange(-90,91,30),
           ylabel='归一化阵列响应 (dB)',title='(a) 全角域方向图')
    zoom.set(xlim=(desired-3,desired+3),ylim=(-1.4,.06),
             ylabel='归一化阵列响应 (dB)',title='(b) 期望方向附近的局部峰')
    zoom.set_xticks(np.arange(desired-3,desired+4))
    title = '筛选展示案例：8 路干扰下的波束方向图' if args.selection == 'showcase' else '最新测试结果：8 路干扰下的波束方向图'
    fig.suptitle(title,fontsize=16,y=.967)
    angle_text=', '.join(f'{v:g}°' for v in interference)
    fig.text(.5,.906,f'期望 AOA：{desired:g}°；干扰 AOA：{angle_text}',ha='center',fontsize=10.5)
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.875),ncol=3,frameon=False,fontsize=10.5)
    fig.text(.07,.213,'场景 SINR：'+ '；'.join(f'{label.split("（")[0]} {metrics[name]["sinr_db"]:.2f} dB'
                                             for name,label in zip(names,labels)),fontsize=10.5)
    fig.text(.07,.174,'最近局部峰指向误差（0.1° 网格）：'+ '；'.join(f'{label.split("（")[0]} {metrics[name]["main_error_deg"]:.2f}°'
                                                      for name,label in zip(names,labels)),fontsize=10.5)
    fig.text(.07,.133,'每条曲线按自身全角域最大功率归一化；绿色虚线为期望方向，灰色点线为干扰方向。',fontsize=9,color='#555555')
    if args.selection == 'showcase':
        fig.text(.07,.092,f'从 {selection["eligible_scenes"]} 个符合条件的 8 路场景中，选取本方法与 MVDR 的 SINR 差距最小者（测试索引 {index}）。',fontsize=9,color='#555555')
        fig.text(.07,.056,'筛选条件：期望角在 ±60° 内，主瓣误差 ≤0.1° 且不大于两对照；该案例用于展示，整体结果见全测试集统计。',fontsize=8.8,color='#555555')
    else:
        fig.text(.07,.082,'取测试集第一个 8 路干扰场景（索引 0），未按预测效果筛选；DNNABF 为当前原始结构复现结果。',fontsize=9,color='#555555')
    basename = 'ch3_selected_eight_interferers' if args.selection == 'showcase' else 'ch3_eight_interferers'
    pdf = out/(basename+'.pdf')
    fig.savefig(pdf,metadata={'Title':'Chapter 3 beam patterns: '+selection['mode'],'Author':'','Subject':'Frozen v11 models; original-form DNNABF reconstruction'})
    fig.savefig(out/(basename+'.png'),dpi=200)
    plt.close(fig)
    np.savez_compressed(out/'pattern_arrays.npz',grid_deg=grid,**curves)
    summary = {'selection_rule':selection,
               'test_index':index,'desired_deg':float(desired),'interference_deg':interference.tolist(),
               'plot_step_deg':.01,'metric_peak_step_deg':.1,
               'normalization':'Each pattern divided by its own maximum full-angle power',
               'metrics':metrics,'source_hashes':{**hashes,str(test):sha256(test),__file__:sha256(__file__)},
               'pdf_sha256':sha256(pdf),
               'scope':'Illustration of one current good-filtered test scene; not the old thesis fixed-angle figure and not a reproduction of original DNNABF accuracy.'}
    save_json(out/'SCENE.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
