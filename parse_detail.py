#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
详细解析 train_backdoor.py 的 output_1.log，提取每组实验的完整统计量。
比 parse_results.py（只报 ASR/BA）更详尽，供 result_all.md 归档 + 论文写作溯源。

日志列: Epoch lr Time TrainLoss TrainACC PoisonLoss PoisonACC(=ASR) CleanLoss CleanACC(=BA) TargetSum CleanSum

用法:
  python parse_detail.py ./results                              # 打印每组详情
  python parse_detail.py ./results results/_detail_summary.json # + 导出 JSON
  python parse_detail.py ./results results/_detail_summary.json --md          # + 打印 markdown 表
  python parse_detail.py ./results results/_detail_summary.json --write-md docs/result_all.md
                                                                # 把 markdown 表按标记覆写进 md 文件
"""
import os, re, sys, json, glob

# ---- 论文 Table 1（CIFAR-10, 1%）四列的 ASR 目标值，供差距对照 ----
# 键: attack ∈ {badnets(Badnets-C), blend(Blended-C), quantizeB(MultiBpp-B), quantize(MultiBpp-RGB)}
# 子键: 选择策略后缀（与 result_dir 命名一致）
PAPER_ALL = {
    "badnets":   {"random": 37.24, "loss": 52.71, "gradient": 52.56, "forget": 71.74,
                  "res_log": 82.13, "res_linear": 68.65, "res_square": 78.76, "res_exp": 76.50},
    "blend":     {"random": 53.41, "loss": 59.43, "gradient": 58.45, "forget": 71.05,
                  "res_log": 82.34, "res_linear": 82.31, "res_square": 84.88, "res_exp": 71.81},
    "quantizeB": {"random": 1.37,  "loss": 28.02, "gradient": 38.26, "forget": 74.39,
                  "res_log": 77.10, "res_linear": 76.73, "res_square": 82.54, "res_exp": 53.92},
    "quantize":  {"random": 1.16,  "loss": 47.85, "gradient": 53.28, "forget": 78.10,
                  "res_log": 80.20, "res_linear": 83.07, "res_square": 83.88, "res_exp": 62.28},
}
ATTACK_LABEL = {"badnets": "Badnets-C", "blend": "Blended-C", "quantizeB": "MultiBpp-B", "quantize": "MultiBpp-RGB"}
MD_START = "<!-- AUTO:result_table_start -->"
MD_END   = "<!-- AUTO:result_table_end -->"

NS_PAT  = re.compile(r"Namespace\((.*)\)")
EPOCH_PAT = re.compile(
    r"^\s*\[(.+?)\]\s*-\s*(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*")


def parse_dirname(name):
    """result_dir 名 → (attack_key, sel_suffix)。sel_suffix 形如 res_square / forget / random ..."""
    for ak in ("quantizeB_", "quantize_", "badnets_", "blend_"):
        if name.startswith(ak):
            return ak.rstrip("_"), name[len(ak):]
    return None, None


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def stdev(xs):
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5


def first_above(rows_asr, thr):
    for ep, asr in rows_asr:
        if asr >= thr:
            return ep
    return -1


def parse_namespace(path):
    ns = None
    with open(path, "r", errors="ignore") as f:
        for line in f:
            m = NS_PAT.search(line)
            if m:
                ns = m.group(1)
    cfg = {}
    if ns:
        for kv in ns.split(", "):
            if "=" in kv:
                k, v = kv.split("=", 1)
                cfg[k.strip()] = v.strip("'\"")
    return cfg


def parse_log(path):
    if not os.path.exists(path):
        return None
    cfg = parse_namespace(path)
    epochs, lrs, times, tr_loss, tr_acc, po_loss, po_acc, cl_loss, cl_acc = [], [], [], [], [], [], [], [], []
    poison_n = None
    with open(path, "r", errors="ignore") as f:
        for line in f:
            m_int = re.match(r"^\s*\[(.+?)\]\s*-\s*(\d+)\s*$", line)
            if m_int and poison_n is None:
                poison_n = int(m_int.group(2))
            m = EPOCH_PAT.match(line)
            if m:
                g = m.groups()
                epochs.append(int(g[1])); lrs.append(float(g[2])); times.append(float(g[3]))
                tr_loss.append(float(g[4])); tr_acc.append(float(g[5]))
                po_loss.append(float(g[6])); po_acc.append(float(g[7]))   # ASR
                cl_loss.append(float(g[8])); cl_acc.append(float(g[9]))   # BA
    if not epochs:
        return None
    n = len(epochs); last = n - 1
    tail20_asr, tail20_ba = po_acc[-20:], cl_acc[-20:]
    tail50_asr = po_acc[-50:] if n >= 50 else po_acc
    best_asr_i = max(range(n), key=lambda i: po_acc[i])
    best_ba_i = max(range(n), key=lambda i: cl_acc[i])
    return {
        "cfg": cfg, "poison_n": poison_n, "n_epochs": n, "final_epoch": epochs[last],
        "time_total_min": round(sum(times) / 60.0, 1), "time_per_epoch_s": round(mean(times), 1),
        "ASR_final": round(po_acc[last] * 100, 2), "ASR_best": round(po_acc[best_asr_i] * 100, 2),
        "ASR_best_epoch": epochs[best_asr_i], "ASR_mean20": round(mean(tail20_asr) * 100, 2),
        "ASR_std20": round(stdev(tail20_asr) * 100, 2), "ASR_mean50": round(mean(tail50_asr) * 100, 2),
        "BA_final": round(cl_acc[last] * 100, 2), "BA_best": round(cl_acc[best_ba_i] * 100, 2),
        "BA_best_epoch": epochs[best_ba_i], "BA_mean20": round(mean(tail20_ba) * 100, 2),
        "BA_std20": round(stdev(tail20_ba) * 100, 2),
        "TrainACC_final": round(tr_acc[last] * 100, 2), "TrainACC_mean20": round(mean(tr_acc[-20:]) * 100, 2),
        "PoisonLoss_final": round(po_loss[last], 4), "CleanLoss_final": round(cl_loss[last], 4),
        "ASR_first_ge50": first_above(list(zip(epochs, po_acc)), 0.5),
        "ASR_first_ge80": first_above(list(zip(epochs, po_acc)), 0.8),
        "lr_milestones": sorted(set(lrs), reverse=True),
    }


def paper_asr(name):
    ak, sel = parse_dirname(name)
    if ak and sel and ak in PAPER_ALL and sel in PAPER_ALL[ak]:
        return PAPER_ALL[ak][sel]
    return None


def fmt_block(name, r):
    c = r["cfg"]
    ak, _ = parse_dirname(name)
    pap = paper_asr(name)
    print("=" * 88)
    print(f"实验组: {name}  [{ATTACK_LABEL.get(ak, ak)}]")
    print(f"  配置: dataset={c.get('dataset')} model={c.get('model')} attack={c.get('backdoor_type')} "
          f"num_levels/type={c.get('num_levels', c.get('type'))}")
    print(f"        selection={c.get('selection')} res_sel={c.get('res_sel')} "
          f"poison_rate={c.get('poison_rate')} y_target={c.get('y_target')} seed={c.get('seed')}")
    print(f"  训练: epochs={r['n_epochs']} 投毒样本数={r['poison_n']} "
          f"总耗时={r['time_total_min']}min ({r['time_per_epoch_s']}s/epoch) lr档={r['lr_milestones']}")
    print(f"  ASR: 末值={r['ASR_final']:.2f} | 末20均={r['ASR_mean20']:.2f}(±{r['ASR_std20']:.2f}) "
          f"| 末50均={r['ASR_mean50']:.2f} | 峰值={r['ASR_best']:.2f}@ep{r['ASR_best_epoch']}")
    print(f"  BA : 末值={r['BA_final']:.2f} | 末20均={r['BA_mean20']:.2f}(±{r['BA_std20']:.2f}) "
          f"| 峰值={r['BA_best']:.2f}@ep{r['BA_best_epoch']}")
    print(f"  训练ACC: 末值={r['TrainACC_final']:.2f} | 末20均={r['TrainACC_mean20']:.2f}")
    print(f"  收敛: PoisonLoss末={r['PoisonLoss_final']} CleanLoss末={r['CleanLoss_final']} "
          f"ASR首破50%@ep{r['ASR_first_ge50']} 首破80%@ep{r['ASR_first_ge80']}")
    if pap is not None:
        print(f"  对照论文: 论文ASR={pap} → 末值ASR差距={r['ASR_final']-pap:+.2f}")
    return r


def emit_md(all_res):
    """生成 result_all.md 的“已完成详细结果”表格（按攻击+ASR_mean20 排序）。"""
    lines = [
        "| 实验组 | 攻击列 | selection | ASR末值 | ASR末20均(±std) | ASR峰值@ep | BA末值 | BA末20均 | TrainACC末 | PoisonLoss末 | 耗时min | 论文ASR | ASR差距 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    def sort_key(kv):
        ak, _ = parse_dirname(kv[0])
        return ({"quantize": 0, "badnets": 1, "blend": 2, "quantizeB": 3}.get(ak, 9), -kv[1]["ASR_mean20"])
    for name, r in sorted(all_res.items(), key=sort_key):
        c = r["cfg"]
        ak, sel = parse_dirname(name)
        sel_disp = c.get("selection", "?") + (f"/{c.get('res_sel')}" if c.get("selection") == "res" else "")
        pap = paper_asr(name)
        gap = f"{r['ASR_final']-pap:+.2f}" if pap is not None else "—"
        pp = f"{pap}" if pap is not None else "—"
        lines.append(
            f"| {name} | {ATTACK_LABEL.get(ak,'?')} | {sel_disp} | {r['ASR_final']:.2f} "
            f"| {r['ASR_mean20']:.2f}(±{r['ASR_std20']:.2f}) | {r['ASR_best']:.2f}@{r['ASR_best_epoch']} "
            f"| {r['BA_final']:.2f} | {r['BA_mean20']:.2f} | {r['TrainACC_final']:.2f} | {r['PoisonLoss_final']} "
            f"| {r['time_total_min']} | {pp} | {gap} |"
        )
    return "\n".join(lines)


def write_md_section(md_path, table_text):
    """把 table_text 覆写进 md_path 中 MD_START...MD_END 之间的区域。"""
    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()
    if MD_START not in content or MD_END not in content:
        print(f"[警告] {md_path} 未找到 {MD_START}/{MD_END} 标记，未写入。")
        return False
    pre = content[:content.index(MD_START) + len(MD_START)]
    post = content[content.index(MD_END):]
    new = pre + "\n" + table_text + "\n" + post
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(new)
    print(f"[已更新] {md_path} 的结果表（{table_text.count(chr(10))-1} 行数据）")
    return True


if __name__ == "__main__":
    resdir = sys.argv[1] if len(sys.argv) > 1 else "./results"
    out_json = sys.argv[2] if len(sys.argv) > 2 else None
    want_md = "--md" in sys.argv
    write_md_path = None
    if "--write-md" in sys.argv:
        i = sys.argv.index("--write-md")
        write_md_path = sys.argv[i + 1] if i + 1 < len(sys.argv) else "docs/result_all.md"

    all_res = {}
    for d in sorted(glob.glob(os.path.join(resdir, "*"))):
        if not os.path.isdir(d):          # 跳过 *.stdout 等非目录
            continue
        name = os.path.basename(d)
        if parse_dirname(name)[0] is None:  # 只解析已知攻击前缀的目录
            continue
        r = parse_log(os.path.join(d, "output_1.log"))
        if r is None:
            print(f"[跳过] {name}: 无有效 epoch 日志")
            continue
        fmt_block(name, r)
        all_res[name] = r

    if out_json:
        with open(out_json, "w") as f:
            json.dump(all_res, f, indent=2, ensure_ascii=False)
        print(f"\n[已导出] {out_json}  共 {len(all_res)} 组")

    md_text = emit_md(all_res)
    if want_md:
        print("\n" + md_text)
    if write_md_path:
        write_md_section(write_md_path, md_text)

    print(f"\n共解析 {len(all_res)} 组实验。")
