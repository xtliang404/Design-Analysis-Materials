import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from bokeh.models import (
    LinearColorMapper, ColorBar, BasicTicker, PrintfTickFormatter,
    HoverTool, FactorRange, ColumnDataSource, Range1d, FixedTicker,
    LabelSet, DataTable, TableColumn, Div
)
from bokeh.plotting import figure, output_file, show
from bokeh.layouts import Spacer, row, column
from bokeh.transform import transform
from factor_analyzer import Rotator

# ================================================
# 1. Load factor_analysis_data.npy
#    Effective data size: 150×17
# ================================================
full_data = np.load("factor_analysis_data.npy", allow_pickle=True)

# Column headers: need1~need17
column_labels = full_data[0, 1:]

# Row headers: Y1~Y150
row_labels = full_data[1:, 0]

# Effective numerical data (150×17)
data = full_data[1:, 1:].astype(float)

# DataFrame (columns=need1~need17, index=Y1~Y150)
df = pd.DataFrame(data, columns=column_labels, index=row_labels)

print("Original rating data (first 10 rows):")
print(df.head(10))

# ================================================
# 2. Standardization and Correlation Matrix
# ================================================
scaler = StandardScaler()
df_standardized = pd.DataFrame(
    scaler.fit_transform(df),
    columns=df.columns,
    index=df.index
)

corr_matrix = df_standardized.corr()
corr_df = corr_matrix.stack().reset_index()
corr_df.columns = ['Variable_X', 'Variable_Y', 'Correlation']
corr_df["label"] = corr_df["Correlation"].apply(lambda x: f"{x:.2f}")
source = ColumnDataSource(corr_df)

# ================================================
# 3. Correlation Heatmap
# ================================================
mapper = LinearColorMapper(palette="RdBu11", low=-1.0, high=1)
hover_heat_map = HoverTool(tooltips=[
    ("Variable X", "@Variable_X"),
    ("Variable Y", "@Variable_Y"),
    ("Correlation", "@Correlation{0.2f}")
])

p = figure(
    title="Correlation Heatmap of Standardized Service Demand Variables",
    x_range=FactorRange(*df.columns),
    y_range=FactorRange(*reversed(df.columns)),
    x_axis_location="below",
    width=900,
    height=850,
    tools=[hover_heat_map, "pan", "wheel_zoom", "box_zoom", "reset", "save"],
    x_axis_label="Demand Variables",
    y_axis_label="Demand Variables"
)

p.rect(
    x="Variable_X", y="Variable_Y", width=1, height=1,
    source=source, fill_color=transform('Correlation', mapper),
    line_color=None
)

p.text(
    x="Variable_X", y="Variable_Y", text="label", source=source,
    text_align="center", text_baseline="middle",
    text_font_size="10pt", text_color="black"
)

# Unified styling
p.title.text_font = "Arial"
p.title.text_font_style = "bold"
p.title.text_font_size = "16pt"
p.xaxis.axis_label_text_font_size = "14pt"
p.yaxis.axis_label_text_font_size = "14pt"
p.xaxis.major_label_text_font_size = "11pt"
p.yaxis.major_label_text_font_size = "11pt"
p.xaxis.axis_label_text_font_style = "normal"
p.yaxis.axis_label_text_font_style = "normal"
p.xaxis.major_label_orientation = np.pi / 4

color_bar = ColorBar(
    color_mapper=mapper,
    ticker=BasicTicker(desired_num_ticks=10),
    formatter=PrintfTickFormatter(format="%.2f"),
    label_standoff=10,
    border_line_color=None,
    location=(0, 0),
    title="Correlation Coefficient",
    title_text_font_style="normal",
    title_text_font_size="12pt",
    major_label_text_font_size="10pt"
)
p.add_layout(color_bar, 'right')

# ================================================
# 4. PCA + Data Preparation
# ================================================
pca = PCA()
pca.fit(df_standardized)

eigenvalues = pca.explained_variance_
explained_var_ratio = pca.explained_variance_ratio_
cumulative_ratio = explained_var_ratio.cumsum()
x_labels = list(range(1, len(eigenvalues) + 1))

pca_source = ColumnDataSource(data=dict(
    x=x_labels,
    eigenvalues=eigenvalues,
    explained_var_ratio=explained_var_ratio,
    cumulative_ratio=cumulative_ratio
))

# ================================================
# 5. Scree Plot 1: Eigenvalues
# ================================================
hover_1 = HoverTool(tooltips=[("Component", "@x"), ("Eigenvalue", "@eigenvalues")])
p1 = figure(
    x_range=Range1d(1, 18),
    y_range=Range1d(0, 10),
    width=700,
    height=350,
    tools=[hover_1, "pan", "wheel_zoom", "box_zoom", "reset", "save"],
    x_axis_label="Principal Component Index",
    y_axis_label="Eigenvalue"
)
p1.line(x='x', y='eigenvalues', source=pca_source, line_width=2, color="green", legend_label="Eigenvalues")
p1.scatter(x='x', y='eigenvalues', source=pca_source, size=8, color="green", marker="circle")
p1.line(x=[x_labels[0], x_labels[-1] + 1], y=[1, 1], color="gray", line_width=2, line_dash="dashed")
p1.line(x=[3, 3], y=[0, 10], color="gray", line_width=2, line_dash="dashed")
p1.legend.location = "top_right"
p1.legend.border_line_color = None

p1.xaxis.major_tick_in = 5
p1.yaxis.major_tick_in = 5
p1.xaxis.major_tick_out = 0
p1.yaxis.major_tick_out = 0

p1.title.text_font = "Arial"
p1.title.text_font_style = "bold"
p1.title.text_font_size = "14pt"
p1.xaxis.axis_label_text_font_size = "12pt"
p1.yaxis.axis_label_text_font_size = "12pt"
p1.xaxis.major_label_text_font_size = "10pt"
p1.yaxis.major_label_text_font_size = "10pt"
p1.xaxis.axis_label_text_font_style = "normal"
p1.yaxis.axis_label_text_font_style = "normal"
p1.legend.label_text_font_size = "10pt"

p1.xaxis.ticker = FixedTicker(ticks=[i for i in range(1, 18)])
p1.yaxis.ticker = FixedTicker(ticks=[i for i in range(0, 10)])
p1.xgrid.visible = False
p1.ygrid.visible = False
p1.outline_line_color = "black"

# ================================================
# 6. Scree Plot 2: Explained Variance
# ================================================
hover_explained = HoverTool(tooltips=[
    ("Component", "@x"),
    ("Explained Variance Ratio", "@explained_var_ratio{0.2f}"),
    ("Cumulative", "@cumulative_ratio{0.2f}")
])

p2 = figure(
    x_range=Range1d(0, 18),
    y_range=Range1d(0, 1),
    width=700,
    height=350,
    tools=[hover_explained, "pan", "wheel_zoom", "box_zoom", "reset", "save"]
)

p2.vbar(x='x', top='explained_var_ratio', source=pca_source, width=0.6, color="orange", legend_label="Explained Variance Ratio")
p2.line(x='x', y='cumulative_ratio', source=pca_source, line_width=2, color="red", legend_label="Cumulative Explained Variance Ratio")
p2.scatter(x='x', y='cumulative_ratio', source=pca_source, size=6, color="red", marker="circle")
p2.line(x=[x_labels[0], x_labels[-1]], y=[0.7, 0.7], color="gray", line_dash="dashed", legend_label="70% Threshold")
p2.legend.location = "center_right"

# Create a label column for bar-top annotations
pca_source.data["explained_var_label"] = [f"{val:.2f}" for val in explained_var_ratio]

labels = LabelSet(
    x='x',
    y='explained_var_ratio',
    text='explained_var_label',
    level='glyph',
    x_offset=0,
    y_offset=3,
    text_align='center',
    text_baseline='bottom',
    text_font_size='10pt',
    source=pca_source
)
p2.add_layout(labels)

p2.title.text_font = "Arial"
p2.title.text_font_style = "bold"
p2.title.text_font_size = "14pt"
p2.xaxis.axis_label = "Principal Component Index"
p2.yaxis.axis_label = "Explained Variance Ratio"
p2.xaxis.axis_label_text_font_size = "12pt"
p2.yaxis.axis_label_text_font_size = "12pt"
p2.xaxis.major_label_text_font_size = "10pt"
p2.yaxis.major_label_text_font_size = "10pt"
p2.xaxis.axis_label_text_font_style = "normal"
p2.yaxis.axis_label_text_font_style = "normal"
p2.legend.label_text_font_size = "10pt"

p2.xaxis.ticker = FixedTicker(ticks=[i for i in range(0, 18)])
p2.yaxis.ticker = FixedTicker(ticks=np.round(np.linspace(0, 1, 11), 2).tolist())
p2.xgrid.visible = False
p2.ygrid.visible = False
p2.outline_line_color = 'black'

p2.xaxis.major_tick_in = 5
p2.yaxis.major_tick_in = 5
p2.xaxis.major_tick_out = 0
p2.yaxis.major_tick_out = 0

# ================================================
# 7. Compute Unrotated Loadings and Varimax-Rotated Loadings
# ================================================
components = pca.components_.T
loadings = components * np.sqrt(pca.explained_variance_)

# Unrotated loading matrix (first 3 components)
loading_df = pd.DataFrame(loadings[:, :3], columns=[f"PC{i+1}" for i in range(3)])
loading_df.insert(0, "Variable", df.columns)
loading_df.to_csv("Unrotated_Loading_Matrix.csv", index=False)

# Varimax rotation using factor_analyzer
rotator = Rotator(method='varimax')
rotated_loadings = rotator.fit_transform(loadings[:, :3])

rotated_loading_df = pd.DataFrame(rotated_loadings, columns=[f"Factor{i+1}" for i in range(3)])
rotated_loading_df.insert(0, "Variable", df.columns)

# Add factor assignment (largest absolute loading)
rotated_loading_df["Factor_Assignment"] = rotated_loading_df.iloc[:, 1:4].abs().idxmax(axis=1)
rotated_loading_df.to_csv("Varimax_Rotated_Loading_Matrix.csv", index=False)

# ================================================
# 8. Output Unrotated/Rotated Loading Matrices as Tables
# ================================================
table1_source = ColumnDataSource(loading_df.round(3))
table2_source = ColumnDataSource(rotated_loading_df.round(3))

columns1 = [TableColumn(field=col, title=col) for col in loading_df.columns]
columns2 = [TableColumn(field=col, title=col) for col in rotated_loading_df.columns]

# DataTable 1
table1 = DataTable(source=table1_source, columns=columns1, width=400, height=500, index_position=None)

# Copyable HTML table 1
table1_html = loading_df.round(3).to_html(index=False, float_format="%.3f", border=1)
table1_div = Div(text=f"{table1_html}", width=400, height=800)

# DataTable 2
table2 = DataTable(source=table2_source, columns=columns2, width=400, height=500, index_position=None)

# Copyable HTML table 2
table2_html = rotated_loading_df.round(3).to_html(index=False, float_format="%.3f", border=1)
table2_div = Div(text=f"{table2_html}", width=400, height=800)

# Table titles
title1 = Div(text="<h3>Unrotated Principal Component Loading Matrix (First 3 Components)</h3>")
title2 = Div(text="<h3>Varimax-Rotated Factor Loading Matrix</h3>")

# ================================================
# 9. Grid Layout: Output All Figures and Tables
# ================================================
layout = column(
    p,
    Spacer(height=50),
    row(p1, p2),
    Spacer(height=40),
    row(column(title1, table1), table1_div, column(title2, table2), table2_div)
)

output_file("Exploratory_Factor_Analysis_Results.html")
show(layout)
