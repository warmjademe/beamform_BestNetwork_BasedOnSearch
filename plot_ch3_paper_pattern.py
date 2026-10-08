"""Paper-size rendering of the already selected frozen case; no reselection."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from beamnas.common import sha256


def main():
    source = Path('output/pdf/ch3_selected_pattern')
    out = Path('output/pdf/ch3_paper_pattern')
    out.mkdir(parents=True, exist_ok=True)
    scene = json.loads((source / 'SCENE.json').read_text())
    with np.load(source / 'pattern_arrays.npz') as data:
        grid = data['grid_deg']
        curves = {key: data[key] for key in
                  ('MVDR_standard', 'ours_optimized', 'DNNABF_standard')}
    plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'font.size': 9,
                         'axes.unicode_minus': False, 'pdf.fonttype': 3,
                         'axes.linewidth': .65, 'xtick.labelsize': 8,
                         'ytick.labelsize': 8, 'axes.labelsize': 9})
    fig, axes = plt.subplots(1, 2, figsize=(6.1, 3.1),
                             gridspec_kw={'width_ratios': [1.65, 1]})
    styles = [('MVDR_standard', 'Oracle MVDR', 'black', '-'),
              ('ours_optimized', 'A-DNNABF', '#0072B2', '--'),
              ('DNNABF_standard', 'DNNABF', '#D55E00', '-.')]
    for ax in axes:
        for key, label, color, ls in styles:
            ax.plot(grid, curves[key], label=label, color=color,
                    linestyle=ls, linewidth=1.05)
        ax.grid(True, linewidth=.4, alpha=.35)
        ax.set_xlabel('扫描角度（°）')
        ax.set_ylabel('归一化功率（dB）')
        ax.tick_params(length=3, width=.6, pad=2)
    desired = scene['desired_deg']
    for theta in scene['interference_deg']:
        axes[0].axvline(theta, color='#777777', linestyle=':', linewidth=.65)
    for ax in axes:
        ax.axvline(desired, color='#009E73', linestyle=':', linewidth=.85)
    axes[0].set(xlim=(-90, 90), ylim=(-100, 2), xticks=np.arange(-90, 91, 30))
    axes[0].set_title('(a) 全角域方向图', fontsize=9)
    axes[1].set(xlim=(desired-3, desired+3), ylim=(-1.4, .06),
                xticks=np.arange(desired-3, desired+4, 2))
    axes[1].set_title('(b) 期望方向局部放大', fontsize=9)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center',
               bbox_to_anchor=(.5, 1.0), ncol=3, frameon=False, fontsize=9)
    fig.subplots_adjust(left=.10, right=.985, bottom=.20, top=.80, wspace=.38)
    path = out / 'ch3_selected_eight_interferers.pdf'
    fig.savefig(path, metadata={'Title': 'Selected eight-interferer test case 1110',
        'Subject': 'Post-test favorable illustration; selection criteria in SCENE.json and thesis caption'})
    fig.savefig(out / 'ch3_selected_eight_interferers.png', dpi=240)
    plt.close(fig)
    provenance = {'test_index': scene['test_index'], 'selection_rule': scene['selection_rule'],
        'normalization': scene['normalization'], 'scope': 'Presentation only; original arrays unchanged',
        'source_array_sha256': sha256(source / 'pattern_arrays.npz'),
        'source_scene_sha256': sha256(source / 'SCENE.json'), 'pdf_sha256': sha256(path)}
    (out / 'PROVENANCE.json').write_text(json.dumps(provenance, indent=2) + '\n')


if __name__ == '__main__':
    main()
