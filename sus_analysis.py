# 导入库
from __future__ import annotations

import argparse
import math
import webbrowser
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, Div, HoverTool
from bokeh.plotting import figure, output_file, save
from bokeh.transform import dodge


# 设置问卷条目和系统名称
ITEM_LABELS = {
    "Q1": "Intention to reuse",
    "Q2": "Interface friendliness",
    "Q3": "System ease of use",
    "Q4": "Learning burden",
    "Q5": "Functional integration",
    "Q6": "Operational consistency",
    "Q7": "Learnability",
    "Q8": "Emotional comfort",
    "Q9": "User confidence",
    "Q10": "Preparation burden",
}
POSITIVE_ITEMS = {"Q1", "Q3", "Q5", "Q7", "Q9"}
INPUT_COLUMNS = ["participant_id"] + list(ITEM_LABELS)
SYSTEMS = ("proposed", "preexisting")


# 读取运行参数和默认文件路径
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze paired ten-item SUS responses for two systems.")
    script_dir = Path(__file__).resolve().parent
    parser.add_argument(
        "proposed_csv",
        nargs="?",
        type=Path,
        default=script_dir / "proposed_system_sus.csv",
    )
    parser.add_argument(
        "preexisting_csv",
        nargs="?",
        type=Path,
        default=script_dir / "preexisting_system_sus.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "sus_analysis_output",
    )
    parser.add_argument("--expected-n", type=int, default=95)
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Do not open the generated HTML report in the default browser.",
    )
    return parser.parse_args()


# 读取单个系统的问卷数据
def load_system_file(path: Path, system: str) -> pd.DataFrame:
    data = pd.read_csv(path, dtype={"participant_id": str})
    missing = [column for column in INPUT_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    result = data[INPUT_COLUMNS].copy()
    result.insert(1, "system", system)
    return result


# 检查样本量、配对关系和量表取值
def validate_input(df: pd.DataFrame, expected_n: int) -> None:
    if df.isna().any().any():
        bad = df.isna().sum()
        raise ValueError(f"Missing values detected:\n{bad[bad > 0]}")
    for item in ITEM_LABELS:
        numeric = pd.to_numeric(df[item], errors="coerce")
        if numeric.isna().any() or not np.all(numeric == np.floor(numeric)):
            raise ValueError(f"{item} contains non-integer responses")
        if not numeric.between(1, 5).all():
            raise ValueError(f"{item} contains responses outside 1-5")
        df[item] = numeric.astype(int)
    duplicate = df.duplicated(["participant_id", "system"], keep=False)
    if duplicate.any():
        raise ValueError("Duplicate participant/system records detected")
    ids = {
        system: set(df.loc[df["system"] == system, "participant_id"])
        for system in SYSTEMS
    }
    for system in SYSTEMS:
        if len(ids[system]) != expected_n:
            raise ValueError(f"{system} has {len(ids[system])}; expected {expected_n}")
    if ids["proposed"] != ids["preexisting"]:
        raise ValueError("The two files do not contain identical participant IDs")


# 将十个SUS条目统一转换为0至4分
def score_items(df: pd.DataFrame) -> pd.DataFrame:
    scored = df.copy()
    for item in ITEM_LABELS:
        if item in POSITIVE_ITEMS:
            scored[f"{item}_score"] = scored[item] - 1
        else:
            scored[f"{item}_score"] = 5 - scored[item]
    return scored


# 计算95%置信区间所需的t临界值
def t_critical_975(df: int) -> float:
    z = NormalDist().inv_cdf(0.975)
    return (
        z
        + (z**3 + z) / (4 * df)
        + (5 * z**5 + 16 * z**3 + 3 * z) / (96 * df**2)
        + (3 * z**7 + 19 * z**5 + 17 * z**3 - 15 * z) / (384 * df**3)
    )


# 计算均值、标准差和95%置信区间
def describe(values: pd.Series) -> dict[str, float | int]:
    clean = values.astype(float)
    n = int(clean.size)
    mean = float(clean.mean())
    sd = float(clean.std(ddof=1))
    half_width = float(t_critical_975(n - 1) * sd / math.sqrt(n))
    return {"n": n, "mean": mean, "sd": sd, "ci_lower": mean - half_width, "ci_upper": mean + half_width}


# 处理Wilcoxon检验中的并列秩
def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = ((start + 1) + end) / 2
        start = end
    return ranks


# 进行配对Wilcoxon符号秩检验并计算效应量
def paired_test(proposed: pd.Series, preexisting: pd.Series) -> dict[str, object]:
    differences = proposed.astype(float).to_numpy() - preexisting.astype(float).to_numpy()
    nonzero = differences[differences != 0]
    n = len(nonzero)
    if n == 0:
        return {
            "n_pairs": len(differences),
            "n_nonzero": 0,
            "zero_method": "wilcox",
            "wilcoxon_w": 0.0,
            "z": 0.0,
            "p_value": 1.0,
            "effect_size_r": 0.0,
            "mean_difference": float(differences.mean()),
        }
    absolute = np.abs(nonzero)
    ranks = average_ranks(absolute)
    w_plus = float(ranks[nonzero > 0].sum())
    w_minus = float(ranks[nonzero < 0].sum())
    expected = n * (n + 1) / 4
    _, counts = np.unique(absolute, return_counts=True)
    tie_correction = sum((t**3 - t) / 48 for t in counts if t > 1)
    variance = n * (n + 1) * (2 * n + 1) / 24 - tie_correction
    delta = w_plus - expected
    continuity = 0.5 * np.sign(delta) if delta else 0.0
    z = float((delta - continuity) / math.sqrt(variance))
    p = float(2 * NormalDist().cdf(-abs(z)))
    return {
        "n_pairs": len(differences),
        "n_nonzero": n,
        "zero_method": "wilcox",
        "wilcoxon_w": min(w_plus, w_minus),
        "z": z,
        "p_value": p,
        "effect_size_r": abs(z) / math.sqrt(n),
        "mean_difference": float(differences.mean()),
    }


# 汇总描述性统计和配对比较结果
def build_outputs(scored: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    item_rows = []
    indexed = {system: scored.loc[scored["system"] == system].set_index("participant_id").sort_index() for system in SYSTEMS}
    for system in SYSTEMS:
        for item, label in ITEM_LABELS.items():
            item_rows.append({"system": system, "item": item, "measure": label, **describe(indexed[system][f"{item}_score"])})
    item_summary = pd.DataFrame(item_rows)
    paired_rows = []
    for item, label in ITEM_LABELS.items():
        paired_rows.append({"item": item, "measure": label, **paired_test(indexed["proposed"][f"{item}_score"], indexed["preexisting"][f"{item}_score"])})
    paired = pd.DataFrame(paired_rows)
    wide = item_summary.pivot(index=["item", "measure"], columns="system", values=["mean", "sd", "ci_lower", "ci_upper"])
    wide.columns = [f"{value}_{system}" for value, system in wide.columns]
    table13 = wide.reset_index().merge(paired, on=["item", "measure"], how="left")
    order = [f"Q{i}" for i in range(1, 11)]
    table13["_order"] = pd.Categorical(table13["item"], order, ordered=True)
    table13 = table13.sort_values("_order").drop(columns="_order")
    return item_summary, paired, table13


# 格式化p值
def format_p_value(value: float) -> str:
    return "<0.001" if value < 0.001 else f"{value:.3f}"


# 保存CSV，文件被占用时继续生成其他结果
def save_csv(data: pd.DataFrame, path: Path, encoding: str = "utf-8") -> None:
    try:
        data.to_csv(path, index=False, encoding=encoding)
    except PermissionError:
        print(f"Warning: {path.name} is open in another program and was not overwritten.")


# 生成论文使用的Table 13
def build_manuscript_table(table13: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for record in table13.to_dict(orient="records"):
        rows.append(
            {
                "Item": str(record["item"]),
                "Evaluation dimension": str(record["measure"]),
                "Proposed mean": f"{record['mean_proposed']:.2f}",
                "Proposed SD": f"{record['sd_proposed']:.2f}",
                "Proposed 95% CI": f"{record['ci_lower_proposed']:.2f}-{record['ci_upper_proposed']:.2f}",
                "Pre-existing mean": f"{record['mean_preexisting']:.2f}",
                "Pre-existing SD": f"{record['sd_preexisting']:.2f}",
                "Pre-existing 95% CI": f"{record['ci_lower_preexisting']:.2f}-{record['ci_upper_preexisting']:.2f}",
                "Mean difference": f"{record['mean_difference']:+.2f}",
                "z": f"{record['z']:.2f}",
                "p": format_p_value(float(record["p_value"])),
                "Effect size r": f"{record['effect_size_r']:.2f}",
            }
        )
    result = pd.DataFrame(rows)
    order = [f"Q{i}" for i in range(1, 11)]
    result["_order"] = pd.Categorical(result["Item"], order, ordered=True)
    return result.sort_values("_order").drop(columns="_order").reset_index(drop=True)


# 使用Bokeh绘制两个系统的条目均值和95%置信区间
def build_bokeh_plot(table13: pd.DataFrame):
    items = table13.copy()
    items["_number"] = items["item"].str.removeprefix("Q").astype(int)
    items = items.sort_values("_number")
    items["dimension"] = items["item"] + "  " + items["measure"]
    source = ColumnDataSource(items)

    plot = figure(
        title="Comparison of direction-aligned SUS item scores",
        y_range=list(reversed(items["dimension"].tolist())),
        x_range=(0, 4),
        width=1250,
        height=560,
        tools="pan,wheel_zoom,box_zoom,reset,save",
        x_axis_label="Direction-aligned item contribution (0-4; higher is better)",
        y_axis_label="SUS item and evaluation dimension",
    )

    plot.segment(
        x0="ci_lower_preexisting",
        y0=dodge("dimension", -0.12, range=plot.y_range),
        x1="ci_upper_preexisting",
        y1=dodge("dimension", -0.12, range=plot.y_range),
        source=source,
        line_width=3,
        color="#627d98",
    )
    preexisting_points = plot.scatter(
        x="mean_preexisting",
        y=dodge("dimension", -0.12, range=plot.y_range),
        source=source,
        size=9,
        color="#627d98",
        marker="circle",
        legend_label="Pre-existing system",
    )
    plot.segment(
        x0="ci_lower_proposed",
        y0=dodge("dimension", 0.12, range=plot.y_range),
        x1="ci_upper_proposed",
        y1=dodge("dimension", 0.12, range=plot.y_range),
        source=source,
        line_width=3,
        color="#d64545",
    )
    proposed_points = plot.scatter(
        x="mean_proposed",
        y=dodge("dimension", 0.12, range=plot.y_range),
        source=source,
        size=9,
        color="#d64545",
        marker="circle",
        legend_label="Proposed system",
    )

    plot.add_tools(
        HoverTool(
            renderers=[preexisting_points],
            tooltips=[
                ("Item", "@item"),
                ("Dimension", "@measure"),
                ("Pre-existing mean", "@mean_preexisting{0.00}"),
                ("95% CI", "@ci_lower_preexisting{0.00}-@ci_upper_preexisting{0.00}"),
            ],
        ),
        HoverTool(
            renderers=[proposed_points],
            tooltips=[
                ("Item", "@item"),
                ("Dimension", "@measure"),
                ("Proposed mean", "@mean_proposed{0.00}"),
                ("95% CI", "@ci_lower_proposed{0.00}-@ci_upper_proposed{0.00}"),
                ("Mean difference", "@mean_difference{+0.00}"),
            ],
        ),
    )
    plot.title.text_font = "Arial"
    plot.title.text_font_style = "bold"
    plot.title.text_font_size = "16pt"
    plot.xaxis.axis_label_text_font_size = "13pt"
    plot.yaxis.axis_label_text_font_size = "13pt"
    plot.xaxis.major_label_text_font_size = "11pt"
    plot.yaxis.major_label_text_font_size = "10pt"
    plot.legend.location = "top_left"
    plot.legend.border_line_color = None
    plot.outline_line_color = "black"
    plot.ygrid.visible = False
    return plot


# 使用Bokeh图形和表格生成HTML文件
def write_table13_html(manuscript_table: pd.DataFrame, statistical_table: pd.DataFrame, output_path: Path) -> None:
    table_html = manuscript_table.to_html(index=False, border=0, justify="center")
    data_table = Div(
        text=(
            "<style>"
            "table{width:100%;border-collapse:collapse;font-family:Arial,sans-serif;font-size:12px;}"
            "th{padding:8px 6px;border-bottom:1px solid #999;text-align:center;}"
            "td{padding:6px;border-bottom:1px solid #ddd;text-align:center;}"
            "tbody tr:nth-child(even){background:#f2f2f2;}"
            "</style>" + table_html
        ),
        width=1450,
        height=345,
    )
    title = Div(
        text="<h1>Table 13. SUS questionnaire results</h1>"
        "<p>Paired evaluation of the pre-existing and proposed systems (n = 95).</p>",
        width=1450,
    )
    note = Div(
        text="<b>Remark:</b> Paired comparisons were conducted using the two-sided Wilcoxon "
        "signed-rank test. Within this test, zero differences were handled using the wilcox method.",
        width=1450,
    )
    layout = column(title, build_bokeh_plot(statistical_table), data_table, note)
    output_file(output_path, title="Table 13. SUS questionnaire results", mode="inline")
    save(layout)


# 运行分析并保存四个结果文件
def main() -> None:
    args = parse_args()
    proposed = load_system_file(args.proposed_csv, "proposed")
    preexisting = load_system_file(args.preexisting_csv, "preexisting")
    data = pd.concat([proposed, preexisting], ignore_index=True)
    validate_input(data, args.expected_n)
    scored = score_items(data)
    item_summary, paired, table13 = build_outputs(scored)
    manuscript_table = build_manuscript_table(table13)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_csv(item_summary, args.output_dir / "item_summary.csv")
    save_csv(paired, args.output_dir / "paired_comparisons.csv")
    table13_csv = args.output_dir / "Table_13_SUS_questionnaire_results.csv"
    table13_html = args.output_dir / "Table_13_SUS_questionnaire_results.html"
    save_csv(manuscript_table, table13_csv, encoding="utf-8-sig")
    write_table13_html(manuscript_table, table13, table13_html)
    print(f"Analysis complete: {args.output_dir.resolve()}")
    print(f"Table 13 CSV: {table13_csv.resolve()}")
    print(f"Table 13 HTML: {table13_html.resolve()}")
    if not args.no_open:
        webbrowser.open(table13_html.resolve().as_uri(), new=2)
        print("Opened Table 13 HTML in the default browser.")


if __name__ == "__main__":
    main()
