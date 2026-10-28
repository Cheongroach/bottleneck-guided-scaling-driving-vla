"""Build the Chinese evidence report from measured JSON and an editable source."""
import csv
import html
import json
from pathlib import Path
import re
from string import Template
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / ".tools/report-deps"))
import mistune
from bs4 import BeautifulSoup
from fontTools import subset
from fontTools.ttLib import TTFont

ASSETS = Path(__file__).resolve().parent
REPORT = ASSETS.parent


def table(headers, rows):
    return "\n".join(["| "+" | ".join(headers)+" |",
                      "| "+" | ".join("---" for _ in headers)+" |"]+
                     ["| "+" | ".join(str(x) for x in row)+" |" for row in rows])


def main():
    e=json.loads((ASSETS/"execution_evidence.json").read_text())
    lookup={r["key"]:r for r in e["matrices"]}
    c=e["calibration"]
    t=e["probe_terminal"]
    values=dict(ref_tf=f'{c["dense_BF16_reference_TF_s"]:.2f}',
                ref_bw=f'{c["copy_effective_GB_s"]/1000:.3f}',
                ridge=f'{c["ridge_algorithm_FLOP_per_byte"]:.2f}',
                probe_sid=e["probe_job"]["sid"],
                probe_wall=f'{t["pod_wall_s"]:.0f}',
                probe_cost=f'{t["GPU_hours"]:.4f}',
                probe_total=f'{t["creation_to_exit_s"]/60:.2f}')
    for token,key,field in (
        ("mlp2_us","b2_mlp_up","warm_kernel_us"),("mlp4_us","b4_mlp_up","warm_kernel_us"),
        ("lora_a_us","b2_lora_a","warm_kernel_us")):
        values[token]=f'{lookup[key][field]["p50"]:.2f}'
    for token,key in (("mlp2_tf","b2_mlp_up"),("mlp4_tf","b4_mlp_up"),
                      ("q2_tf","b2_q_forward"),("kv2_tf","b2_kv_forward"),
                      ("lora_a_tf","b2_lora_a"),("lora_b_tf","b2_lora_b")):
        values[token]=f'{lookup[key]["warm_kernel_TF_s"]:.2f}'
    values["lora_a_event"]=f'{lookup["b2_lora_a"]["warm_event_mean_ms"]*1000:.2f}'
    selected=["b2_q_forward","b2_q_dx","b2_kv_forward","b2_mlp_up","b2_mlp_up_dx",
              "b2_mlp_down","b2_lora_a","b2_lora_b","b4_mlp_up","b8_mlp_up"]
    values["matrix_table"]=table(
        ["Shape","Warm kernel μs","Cold kernel μs","Warm TFLOP/s","Warm event μs"],
        [[key,f'{lookup[key]["warm_kernel_us"]["p50"]:.2f}',
          f'{lookup[key]["cold_kernel_us"]["p50"]:.2f}',
          f'{lookup[key]["warm_kernel_TF_s"]:.2f}',
          f'{lookup[key]["warm_event_mean_ms"]*1000:.2f}'] for key in selected])
    values["resource_table"]=table(["代表 kernel","Grid / CTA 数","Threads/block","Regs/thread","Shared KiB/block"],
        [[key, str(r["grid"])+" / "+str(r["grid"][0]*r["grid"][1]*r["grid"][2]),
          r["block"][0]*r["block"][1]*r["block"][2],r["registers per thread"],
          f'{r["shared memory"]/1024:.1f}'] for key in
         ["b2_mlp_up","b2_lora_a","b4_lora_a","dense_reference"]
         for r in [lookup[key]["kernel_samples"][1]["launch"]]])
    values["communication_table"]=table(
        ["Rank0 trace","步数","Gradient NCCL/步","All NCCL/步","与计算交集/步","NCCL 独占区间/步"],
        [[key+" 卡",v["steps"],f'{v["gradient_nccl_union_ms"]/v["steps"]:.3f} ms',
          f'{v["nccl_union_ms"]/v["steps"]:.3f} ms',
          f'{v["nccl_compute_overlap_ms"]/v["steps"]:.3f} ms',
          f'{v["nccl_only_ms"]/v["steps"]:.3f} ms'] for key in ["16","32"]
         for v in [e["traces"][key]]])
    values["budget_table"]=table(
        ["每卡 B","Physical M","Linear forward TF","Dense attn forward TF","NoCP Linear TF","CP Linear TF 范围"],
        [[v["batch"],v["batch"]*256,f'{v["linear_forward_GF"]/1000:.3f}',
          f'{v["dense_attention_forward_GF"]/1000:.4f}',
          f'{v["linear_train_without_cp_approx_GF"]/1000:.3f}',
          "–".join(f'{x/1000:.3f}' for x in v["linear_train_with_cp_range_GF"])]
         for v in e["model"]["budgets"]])
    source=Template((ASSETS/"report_source.md").read_text()).substitute(values)
    chart_pattern=r"\[\[CHART:([^|]+)\|([^|]+)\|([^\]]+)\]\]"
    charts={m[0]:dict(title=m[1],note=m[2]) for m in re.findall(chart_pattern,source)}
    md=re.sub(chart_pattern,lambda m:f'![{m[2]}](assets/{m[1]}.svg)\n\n{m[3]}',source)
    (REPORT/"execution-roofline.md").write_text(md)
    renderer=mistune.create_markdown(escape=False,plugins=["table"])
    soup=BeautifulSoup(renderer(md),"html.parser")
    soup.h1.decompose()
    soup.find("p").decompose()
    for tab in soup.find_all("table"):
        wrapper=soup.new_tag("div",attrs={"class":"table-wrap"})
        tab.wrap(wrapper)
    for name,info in charts.items():
        img=soup.find("img",src=f"assets/{name}.svg")
        fig=soup.new_tag("figure",attrs={"class":"chart-figure"})
        cap=soup.new_tag("figcaption"); cap.string=info["title"]; fig.append(cap)
        div=soup.new_tag("div",id=f"chart-{name}",attrs={"class":"chart-container"})
        fig.append(div)
        no=soup.new_tag("noscript"); fallback=soup.new_tag("img",src=f"assets/{name}.svg",alt=info["title"])
        no.append(fallback);fig.append(no)
        small=soup.new_tag("p",attrs={"class":"figure-note"});small.string=info["note"];fig.append(small)
        link=soup.new_tag("a",href=f"assets/{name}.svg",attrs={"class":"figure-download"})
        link.string="打开独立 SVG";fig.append(link)
        parent=img.parent
        following=parent.find_next_sibling()
        if following and following.name=="p" and following.get_text().strip()==info["note"]:
            following.decompose()
        parent.replace_with(fig)
    for code in soup.select("code.language-mermaid"):
        code.parent["class"]=["mermaid"]
        code.parent.string=code.get_text()
    nav=[]
    for i,h2 in enumerate(soup.find_all("h2"),1):
        h2["id"]=f"s{i:02}"
        nav.append(f'<a href="#s{i:02}">{html.escape(h2.get_text().split("：")[0])}</a>')
    text=soup.get_text()+" ".join(x["title"] for x in charts.values())+\
         "计算强度 FLOP/byte 逻辑吞吐 TFLOP/s 相对时间 ms CPU forward backward optimizer GPU NCCL gradient tail 平均 快步 慢步比例 CP on off 经验参考 Warm Cold Kernel Event"
    font=TTFont("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",fontNumber=0)
    sub=subset.Subsetter();sub.populate(text=text);sub.subset(font)
    font.save(REPORT/"_shared/fonts/WenQuanYi-subset.ttf")
    data=json.loads((ASSETS/"chart_data.json").read_text())
    decomposition=next(v for v in e["history"]["distribution_decomposition"] if v["threshold_ms"]==250)
    data["scaling"]=[dict(world=int(k),mean_ms=e["history"]["runs"][k]["mean_step_ms"],
                         fast_ms=v["fast_mean_ms"],slow_fraction=1-v["fast_fraction"])
                     for k,v in decomposition["runs"].items()]
    with (ROOT/"results/gc_probe_steps.csv").open() as stream:
        rows=list(csv.DictReader(stream))
    data["gc_steps"]={}
    for key in ["ckpton_gcauto","ckpton_gcdisabled","ckptoff_gcauto"]:
        selected=[r for r in rows if r["key"]==key and int(r["repeat"])==0]
        assert len(selected)==96
        data["gc_steps"][key]=[dict(step=int(r["step"]),forward_ms=float(r["forward_ms"]),
                                   gc_ms=float(r["forward_gc_ms"]),
                                   gc_gen2="2" in r["forward_gc_generations"].split(",")) for r in selected]
    data["chart_titles"]=charts
    (ASSETS/"chart_data.json").write_text(json.dumps(data,ensure_ascii=False,indent=2))
    (ASSETS/"chart_data.js").write_text("window.executionChartData="+json.dumps(data,ensure_ascii=False)+";\n")
    css=(ASSETS/"report.css").read_text()
    doc=f"""<!-- Generated by Trae Work -->
<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Driving-VLA · 执行流程与 Roofline 证据</title><style>{css}</style></head>
<body><article class="page"><header>
<p class="eyebrow">DRIVING-VLA / H20 / PERFORMANCE EVIDENCE</p>
<h1>从一次更新<br>到 Roofline</h1>
<p class="subtitle">Forward / backward 耗时、延迟隐藏、通信与计算强度</p>
<p class="meta">2026-10-28 · 历史训练 + 单卡因果对照 + 25 组矩阵微测</p>
<div class="facts"><div><strong>−8.62%</strong><span>32 卡吞吐变化</span></div>
<div><strong>2.035×</strong><span>单卡 B2 关闭 CP 收益</span></div>
<div><strong>{values["ref_tf"]}</strong><span>实测 BF16 参考 TFLOP/s</span></div></div>
<p class="reading">阅读顺序：先看 01–02 的执行直觉，再看 03–06 的实测流程，最后用 07–10 的公式、边界与复核入口深入验证。</p>
<div class="downloads"><a href="execution-roofline.md">Markdown</a><a href="assets/execution_evidence.json">证据 JSON</a><a href="assets/matrix_measurements.csv">矩阵 CSV</a></div>
</header><nav aria-label="报告目录">{''.join(nav)}</nav><main>{soup}</main>
<footer>本报告可离线阅读；图表数据、脚本和字体均在本目录中。外部链接仅用于方法文献。</footer>
</article><script src="./_shared/js/echarts.min.js"></script>
<script src="./assets/chart_data.js"></script><script src="./assets/charts.js"></script></body></html>"""
    (REPORT/"execution-roofline.html").write_text(doc)
    print(json.dumps(dict(chapters=len(nav),charts=len(charts),markdown_chars=len(md),
                          CJK_font_bytes=(REPORT/"_shared/fonts/WenQuanYi-subset.ttf").stat().st_size)))


if __name__=="__main__":
    main()
