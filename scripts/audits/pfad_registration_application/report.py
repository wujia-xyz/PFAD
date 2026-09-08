"""QA and report for the frozen retrospective registration application."""
from pathlib import Path
import json, csv, hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'artifacts/pfad_registration_application_20260906'
METHODS=('raw','random_matched_refined','geometry_matched_refined','pfad_refined')
NAMES={'raw':'Raw','random_matched_refined':'Random matched','geometry_matched_refined':'Geometry matched','pfad_refined':'PFAD'}


def main():
    d=json.loads((OUT/'summary.json').read_text());protocol=json.loads((OUT/'protocol.json').read_text())
    rows=[];matched=True;pred_hashes=True;protocol_hashes=True
    h=hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest()
    for path in sorted((OUT/'cases').glob('*.json')):
        case=json.loads(path.read_text());rs=case['records'];assert len(rs)==64
        by_start={}
        for r in rs:by_start.setdefault((r['condition'],r['draw']),[]).append(r)
        assert len(by_start)==16
        for group in by_start.values():
            assert {r['method'] for r in group}==set(METHODS)
            assert all(np.array_equal(group[0]['imposed_transform'],r['imposed_transform']) for r in group)
            for r in group:
                T=np.array(r['estimated_transform']);assert np.isfinite(T).all()
                assert np.allclose(T[3],[0,0,0,1])
                assert np.allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=1e-5)
        src=ROOT/'results/first_arrival_refinement/rebuild_v2/selector_attentive_loocv/nested_policy'/f"{case['specimen']}_{case['anatomy']}.npz"
        pred_hashes &= hashlib.sha256(src.read_bytes()).hexdigest()==case['input_prediction_sha256']
        protocol_hashes &= case['protocol_sha256']==h
        assert case['metadata']['counts_matched_per_record']
        rows.extend(rs)
    assert len(rows)==2688 and pred_hashes and protocol_hashes
    flat=[{k:v for k,v in r.items() if k not in ['imposed_transform','estimated_transform']} for r in rows]
    with (OUT/'all_trials.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    strata={}
    for condition in ['aligned','5deg_5mm','10deg_10mm','20deg_20mm']:
        strata[condition]={m:{k:float(np.mean([r[k] for r in rows if r['condition']==condition and r['method']==m])) for k in ['virtual_target_rms_displacement_mm','registration_recall']} for m in METHODS}
    (OUT/'initialization_strata.json').write_text(json.dumps(strata,indent=2))
    qa={'cases':42,'all_trials':2688,'primary_trials':2520,'identity_start_diagnostics':168,
        'paired_initializations_identical':True,'all_transforms_finite_rigid':True,
        'all_input_prediction_hashes_unchanged':pred_hashes,'all_protocol_hashes_match':protocol_hashes,
        'numerical_exceptions':sum(r['error'] is not None for r in rows),'zero_final_fitness':sum(r['fitness']==0 for r in rows),
        'independent_statistical_units':14,'selection_inputs_use_no_CT_visibility_filter':True}
    (OUT/'QA.json').write_text(json.dumps(qa,indent=2))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(7.16,3.05))
    for ax,key,title,scale in [(axes[0],'virtual_target_rms_displacement_mm','(a) Virtual-target RMS displacement',1),(axes[1],'registration_recall','(b) Registration success',100)]:
        a=np.array(d['specimen_values']['raw'][key])*scale;b=np.array(d['specimen_values']['pfad_refined'][key])*scale
        for x,y in zip(a,b):ax.plot([0,1],[x,y],color='#CAD2D8',lw=.8,zorder=1)
        ax.scatter(np.zeros(14),a,facecolors='white',edgecolors='#536674',s=28,label='Raw',zorder=3)
        ax.scatter(np.ones(14),b,color='#176F80',marker='^',s=31,label='PFAD',zorder=3)
        ax.set_xticks([0,1],['Raw','PFAD']);ax.set_xlim(-.3,1.3);ax.set_title(title,loc='left',fontsize=10)
        ax.set_ylabel('mm' if scale==1 else '%');ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'paired_registration.pdf');fig.savefig(OUT/'paired_registration.png',dpi=300);plt.close(fig)
    lines=['# PFAD下游配准应用实验：完整结果','',
      '已完成现有真实数据上的回顾性配准恢复实验，不需要新采集。协议在读取配准结果前固定，全部结果保留。',
      '', '## 实验内容','',
      '- 14个标本、足/胫骨/腓骨共42病例。',
      '- 4种输入：原始候选、匹配点数的随机选择、匹配点数的跨扫描近邻选择、冻结PFAD。后三者均使用相同已冻结深度校正。',
      '- 配准器固定为多尺度Huber point-to-plane ICP，目标为完整的对应CT骨表面；不得用CT可见性筛选输入。',
      '- 初始错位为旋转/平移幅值5°/5mm、10°/10mm、20°/20mm，每档5个随机方向；另有无错位诊断。各方法使用相同错位。',
      '- 共2688次配准，主要分析2520次，另168次对齐起点诊断。统计先平均各起点和解剖，再以14标本配对，不能把重复初始化当成独立标本。',
      '- 主要终点为相对数据集原始配准关系的虚拟靶点RMS位移及配准成功率。没有独立手术靶点，因此不称临床TRE。成功阈值RTE<4mm且RRE<5°仅为计算基准判据。',
      '- CT仅供下游配准与评价使用，PFAD选点已冻结。此实验不意味着整个手术工作流无CT。',
      '', '## 预先指定的总体结果','',
      '|输入点集|虚拟靶点RMS位移/mm|平移误差/mm|旋转误差/°|成功率|','|---|---:|---:|---:|---:|']
    for m in METHODS:
        x=d['summary'][m];lines.append(f"|{NAMES[m]}|{x['virtual_target_rms_displacement_mm']['mean']:.3f}|{x['translation_error_mm']['mean']:.3f}|{x['rotation_error_deg']['mean']:.3f}|{100*x['registration_recall']['mean']:.2f}%|")
    lines += ['', 'PFAD相对Raw的平均RMS改善为1.019mm，95%标本bootstrap区间[-0.017,2.382]mm；配对Wilcoxon原始p=0.268，6项主检验Holm后p=1.000。成功率仅提高1.75个百分点，其Holm后p也为1.000。相对随机/几何匹配对照的两个主要终点同样未达到预设的多重比较后显著性。不能将均值方向写成显著应用收益。',
      '', '## 解剖分层（描述性，不替代总体结论）','',
      '|解剖|Raw RMS/mm|PFAD RMS/mm|Raw成功率|PFAD成功率|','|---|---:|---:|---:|---:|']
    for anatomy,x in d['by_anatomy_descriptive'].items():
        a=x['raw'];b=x['pfad_refined'];lines.append(f"|{anatomy}|{a['virtual_target_rms_displacement_mm']:.3f}|{b['virtual_target_rms_displacement_mm']:.3f}|{100*a['registration_recall']:.2f}%|{100*b['registration_recall']:.2f}%|")
    lines += ['', '腓骨描述性改善较大；胫骨的RMS与成功率均未改善。总体RMS在8/14个标本改善、6/14恶化。该不均匀性限制了稳健应用收益的主张。未按这些结果重新调参、删病例或更换主要终点。',
      '', '## 如何用于论文','',
      '现阶段可以报告为探索性的下游配准结果，并明确总体未显著；它不足以把“显著提升配准/导航”增加为论文贡献，也不足以承诺JBHI录用。结果还没有写入现有主稿，以免在用户审阅应用方向前改变主论文的定位。若继续分析，应利用已保留的全部变换检查失败模式，而不是选择有利的初值、解剖或成功阈值替代本次主结论。',
      '', '新采集不可行时，可以维持公开离体数据的研究范围，并把独立采集泛化作为局限。这项回顾性应用验证与独立采集验证回答不同问题。',
      '', '## 文件与复现','',
      '- protocol.json：预先冻结协议。',
      '- cases/：42病例的逐次变换、指标、输入和协议哈希。',
      '- all_trials.csv / summary.json：完整数值与预定统计。',
      '- initialization_strata.json：包含对齐起点的全部预定初值分层。',
      '- paired_registration.pdf/png：14标本配对图。',
      '- QA.json / numerical_sanity.json：输入、配对初值、刚性变换和指标约定检查。',
      '- 脚本：scripts/audits/pfad_registration_application/run.py及report.py。',
      '', '执行时OMP_NUM_THREADS=1、OPENBLAS_NUM_THREADS=1、4个CPU工作进程。记录的时间仅为配准阶段，排除PFAD训练、源特征构造、磁盘读取和目标预处理，不可当作完整系统耗时。',
      '', '相关先例：https://github.com/luohwu/NeuralBoneReg-implementation。该项目也使用UltraBones100k和已知刚性错位恢复；本次采用不同初值范围和固定ICP后端，不做跨协议数值排名。']
    (OUT/'RESULTS_ZH.md').write_text('\n'.join(lines)+'\n')
    (OUT/'COMPLETION.json').write_text(json.dumps({'status':'complete','cases':42,'registration_runs':2688,
        'statistically_significant_primary_application_advantage':False,'new_acquisitions':0,
        'clinical_TRE_measured':False,'manuscript_modified':False},indent=2))
    print(json.dumps(qa),flush=True)


if __name__=='__main__':main()
