#!/usr/bin/env python3
"""Publication figure and a compact exact-lookup table from complete results."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager,ticker
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT / "src"))
from pfad_tcsvt.common import dump_json,experiment_directory,sha256

COLORS={"raw":"#CF7455","cross_sweep_nearest":"#8394A2","set_transformer":"#3F4851","pfad":"#176F86"}
LABELS={"raw":"Raw","cross_sweep_nearest":"Nearest","set_transformer":"Set Transformer","pfad":"PFAD"}
STYLES={"raw":(":","x"),"cross_sweep_nearest":("--","^"),"set_transformer":("-.","s"),"pfad":("-","o")}
METHODS=tuple(COLORS)


def format_p(value):
    return r"$<0.001$" if value<.001 else f"{value:.3f}"


def table_text(data):
    metrics=("chamfer_l1_mm","hd95_mm","fscore_1mm","normal_consistency")
    lines=[r"\begin{table*}[t]",r"\centering",r"\caption{Comparison With the Set Transformer Support Predictor}",
           r"\label{tab:set_transformer}",r"\setlength{\tabcolsep}{7pt}",r"\begin{tabular}{lrrrrrrrr}",r"\toprule",
           r" & \multicolumn{4}{c}{Point set} & \multicolumn{4}{c}{Open surface} \\",
           r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
           r"Method & CD (mm)$\downarrow$ & HD95 (mm)$\downarrow$ & F1$\uparrow$ & NC$\uparrow$ & CD (mm)$\downarrow$ & HD95 (mm)$\downarrow$ & F1$\uparrow$ & NC$\uparrow$ \\",r"\midrule"]
    for method,label in [("set_transformer","Set Transformer"),("pfad",r"\PFAD")]:
        values=[data["clean_comparator"][rep][method][m]["mean"] for rep in ["points","meshes"] for m in metrics]
        lines.append(label+" & "+" & ".join(f"{v:.3f}" for v in values)+r" \\")
    ps=[data["clean_comparator_tests"][rep][m]["holm_adjusted_p"] for rep in ["points","meshes"] for m in metrics]
    lines.extend([r"$\pholm$"+" & "+" & ".join(format_p(p) for p in ps)+r" \\",r"\bottomrule",r"\end{tabular}",r"\end{table*}"])
    return "\n".join(lines)+"\n"


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",type=Path,default=ROOT / "configs/pfad/tcsvt/acquisition_robustness_v1.json")
    parser.add_argument("--output-dir",type=Path,default=ROOT / "outputs/figures")
    args=parser.parse_args()
    _,run=experiment_directory(args.config)
    summary=ROOT / "paper_results/acquisition_summary.json"
    data=json.loads(summary.read_text())
    if data["status"]!="complete" or data["quality_checks"]["cases"]!=42:raise RuntimeError("complete cohort required")
    out=args.output_dir
    out.mkdir(parents=True,exist_ok=True)
    font_paths=[Path('/usr/share/fonts/truetype/croscore/Arimo-Regular.ttf'),Path('/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf')]
    family="DejaVu Sans"
    for path in font_paths:
        if path.exists():
            font_manager.fontManager.addfont(str(path));family=font_manager.FontProperties(fname=str(path)).get_name();break
    plt.rcParams.update({"font.family":family,"font.size":9,"axes.labelsize":9.5,"axes.titlesize":9.5,
                        "xtick.labelsize":9,"ytick.labelsize":9,"legend.fontsize":9,"pdf.fonttype":42,
                        "ps.fonttype":42,"svg.fonttype":"none","axes.linewidth":.65,
                        "text.color":"#233346","axes.labelcolor":"#233346","xtick.color":"#233346","ytick.color":"#233346"})
    size_inches=(3.5,4.2)
    fig,axes=plt.subplots(3,2,figsize=size_inches,sharey="col")
    fig.subplots_adjust(left=.115,right=.985,bottom=.11,top=.825,wspace=.34,hspace=.70)
    groups=[("(a) Sweep availability",["clean:0","sweeps:0.75","sweeps:0.5"],[100,75,50],"Sweep budget (%)"),
            ("(b) Frame sampling",["clean:0","frames:2","frames:4"],[1,2,4],"Frame stride"),
            ("(c) Pose error",["clean:0","pose:0.5","pose:1.0","pose:2.0"],[0,.5,1,2],"Magnitude (° and mm)")]
    values_manifest=[]
    handles=[]
    metrics=["chamfer_l1_mm","fscore_1mm"]
    for col,metric in enumerate(metrics):
        all_limits=[]
        for row,(title,keys,x,xlabel) in enumerate(groups):
            ax=axes[row,col]
            for method in METHODS:
                statistics=[data["acquisition"][key][method][metric] for key in keys]
                y=np.array([s["mean"] for s in statistics]);lo=np.array([s["mean_95ci"][0] for s in statistics]);hi=np.array([s["mean_95ci"][1] for s in statistics])
                all_limits.extend(lo.tolist()+hi.tolist())
                line,marker=STYLES[method]
                trace,=ax.plot(x,y,color=COLORS[method],linestyle=line,marker=marker,linewidth=1.15,
                    markersize=4.3 if method=="set_transformer" else 3.7,markeredgewidth=.8,
                    markerfacecolor="white" if method in ["cross_sweep_nearest","set_transformer"] else COLORS[method],label=LABELS[method])
                ax.fill_between(x,lo,hi,color=COLORS[method],alpha=.075,linewidth=0)
                if row==0 and col==0:handles.append(trace)
                values_manifest.append({"metric":metric,"method":method,"condition_keys":keys,"x":x,"means":y.tolist(),"ci_low":lo.tolist(),"ci_high":hi.tolist()})
            ax.set_xticks(x)
            ax.set_xticklabels([f"{v:g}" for v in x])
            if row==0:ax.set_xlim(104,46)
            else:
                margin=(max(x)-min(x))*.08;ax.set_xlim(min(x)-margin,max(x)+margin)
            ax.grid(axis="y",color="#E6EBEF",linewidth=.45)
            ax.set_axisbelow(True)
            ax.spines[["top","right"]].set_visible(False)
            ax.spines[["left","bottom"]].set_color("#A9B3BD")
            ax.tick_params(length=2.5,width=.65,pad=2)
        span=max(all_limits)-min(all_limits)
        lower=min(all_limits)-.08*span;upper=max(all_limits)+.08*span
        if metric=="fscore_1mm":lower=max(0,lower);upper=min(1,upper)
        for ax in axes[:,col]:
            ax.set_ylim(lower,upper)
            ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f" if col==0 else "%.2f"))
    axes[0,0].set_title("Chamfer-L1 (mm)",pad=5)
    axes[0,1].set_title("F1 at 1 mm",pad=5)
    row_labels=["(a) Sweep budget (%)","(b) Frame stride","(c) Pose error (° and mm)"]
    for row,label in enumerate(row_labels):
        left=axes[row,0].get_position();right=axes[row,1].get_position()
        fig.text((left.x0+right.x1)/2,left.y0-.073,label,ha="center",va="center",fontsize=9.5)
    fig.legend(handles,[LABELS[m] for m in METHODS],loc="upper center",bbox_to_anchor=(.52,.995),ncol=2,frameon=False,
               handlelength=2.0,columnspacing=1.25,handletextpad=.4,borderaxespad=.15,labelspacing=.45)
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    bounds=[]
    for artist in fig.findobj(matplotlib.text.Text):
        if artist.get_visible() and artist.get_text():
            bbox=artist.get_window_extent(renderer)
            if bbox.width and bbox.height:
                bounds.append({"text":artist.get_text(),"bounds_px":list(bbox.bounds)})
    for extension in ["pdf","svg"]:fig.savefig(out/f"fig8_acquisition_sensitivity.{extension}",facecolor="white")
    fig.savefig(out/"fig8_acquisition_sensitivity.png",dpi=600,facecolor="white")
    plt.close(fig)
    (out/"table_set_transformer.tex").write_text(table_text(data))
    dump_json(out/"source_data.json",{"source_summary":str(summary.relative_to(ROOT)),"summary_sha256":sha256(summary),
        "code_sha256":sha256(Path(__file__)),"font_family":family,"size_inches":list(size_inches),
        "layout":{"rows":"acquisition condition families","columns":metrics,"intended_width":"one_column","minimum_text_pt":9},
        "plotted_series":values_manifest,"text_bounds":bounds})
    print(json.dumps({"figure":str(out/"fig8_acquisition_sensitivity.pdf"),"table":str(out/"table_set_transformer.tex")}),flush=True)


if __name__=="__main__":main()
