"""
Data visualization module creating responsive Plotly charts.
Follows healthcare analytics theme with clean layouts, legible fonts, and clear labels.
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from config.settings import THEME
from src.preprocessing import format_inr_amount, usd_to_inr


def style_figure(fig: go.Figure, title: str = "", height: int = 380) -> go.Figure:
    """Applies consistent corporate styling to a Plotly figure."""
    fig.update_layout(
        title={
            "text": f"<b>{title}</b>",
            "x": 0.02,
            "y": 0.95,
            "xanchor": "left",
            "yanchor": "top",
            "font": {"size": 15, "color": THEME["secondary"]}
        },
        template="plotly_white",
        height=height,
        margin=dict(l=40, r=30, t=50, b=40),
        font=dict(family="Inter, Segoe UI, sans-serif", size=12, color="#334155"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(bgcolor="white", font_size=12, font_family="Inter, sans-serif")
    )
    return fig


def format_inr_axis(fig: go.Figure, values: pd.Series, axis: str = "y") -> go.Figure:
    """Formats INR chart ticks with Indian digit grouping."""
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return fig
    low, high = float(numeric.min()), float(numeric.max())
    ticks = np.linspace(low, high, 5).tolist() if low != high else [low]
    tick_labels = [format_inr_amount(value) for value in ticks]
    if axis == "x":
        fig.update_xaxes(tickmode="array", tickvals=ticks, ticktext=tick_labels)
    elif axis == "y":
        fig.update_yaxes(tickmode="array", tickvals=ticks, ticktext=tick_labels)
    else:
        raise ValueError(f"Unsupported chart axis: {axis}")
    return fig


def plot_claim_type_donut(inpatient_count: int, outpatient_count: int) -> go.Figure:
    """Creates a donut chart comparing inpatient vs outpatient claim volumes."""
    labels = ["Outpatient Claims", "Inpatient Admissions"]
    values = [outpatient_count, inpatient_count]
    colors = [THEME["accent"], THEME["primary"]]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.6,
        marker=dict(colors=colors, line=dict(color="#FFFFFF", width=2)),
        textinfo="label+percent",
        hoverinfo="label+value+percent"
    )])
    return style_figure(fig, "Claim Volume Distribution by Setting", height=320)


def plot_provider_risk_tiers(risk_counts: Dict[str, int]) -> go.Figure:
    """Creates a bar chart displaying provider distribution across Low, Medium, High risk tiers."""
    order = ["Low", "Medium", "High"]
    counts = [risk_counts.get(k, 0) for k in order]
    colors = [THEME["success"], THEME["warning"], THEME["danger"]]

    fig = go.Figure(data=[go.Bar(
        x=order,
        y=counts,
        text=[f"{v:,}" for v in counts],
        textposition="auto",
        marker=dict(color=colors, line=dict(color="#CBD5E1", width=1))
    )])
    fig.update_xaxes(title="Risk Tier")
    fig.update_yaxes(title="Number of Providers", gridcolor="#F1F5F9")
    return style_figure(fig, "Provider Risk Tier Classification", height=320)


def plot_model_comparison_radar_or_bar(comparison_df: pd.DataFrame) -> go.Figure:
    """Creates a grouped bar chart comparing RF, LR, and XGBoost metrics."""
    metrics = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC", "Average Precision"]
    available_metrics = [m for m in metrics if m in comparison_df.columns]

    fig = go.Figure()
    colors = [THEME["primary"], THEME["accent"], "#6366F1"]

    for idx, row in comparison_df.iterrows():
        fig.add_trace(go.Bar(
            name=row["Model"],
            x=available_metrics,
            y=[row[m] for m in available_metrics],
            marker_color=colors[idx % len(colors)]
        ))

    fig.update_layout(barmode="group")
    fig.update_yaxes(range=[0, 1.05], tickformat=".0%", gridcolor="#F1F5F9")
    return style_figure(fig, "Cross-Validated Model Comparison (5-Fold Stratified CV)", height=380)


def plot_shap_contributions(contributions: List[Dict[str, Any]]) -> go.Figure:
    """Creates a horizontal waterfall/contribution bar chart for local SHAP values."""
    df = pd.DataFrame(contributions).sort_values("shap_value", ascending=True)

    colors = [THEME["danger"] if v > 0 else THEME["success"] for v in df["shap_value"]]

    fig = go.Figure(go.Bar(
        x=df["shap_value"],
        y=df["display_name"],
        orientation="h",
        marker=dict(color=colors)
    ))
    fig.update_xaxes(title="SHAP Value (Impact on Fraud Risk Log-Odds)", zeroline=True, zerolinecolor="#64748B")
    return style_figure(fig, "Local Feature Contributions to Prediction", height=380)


def plot_beneficiary_claim_timeline(claims_df: pd.DataFrame) -> go.Figure:
    """Creates an interactive scatter/timeline chart of claims submitted for a beneficiary."""
    if claims_df.empty:
        return go.Figure()

    plot_df = claims_df.copy()
    plot_df["ReimbursementINR"] = pd.to_numeric(
        plot_df["InscClaimAmtReimbursed"], errors="coerce"
    ).map(usd_to_inr)
    plot_df["ReimbursementDisplayINR"] = plot_df["ReimbursementINR"].map(
        format_inr_amount
    )
    plot_df["SourceReimbursementUSD"] = pd.to_numeric(
        plot_df["InscClaimAmtReimbursed"], errors="coerce"
    )
    fig = px.scatter(
        plot_df,
        x="ClaimStartDt",
        y="ReimbursementINR",
        color="ClaimType",
        size="ReimbursementINR",
        hover_data=[
            "ClaimID", "Provider", "AIReviewRecommendation", "ClaimDurationDays",
            "ReimbursementDisplayINR", "SourceReimbursementUSD",
        ],
        color_discrete_map={"Inpatient": THEME["primary"], "Outpatient": THEME["accent"]},
        labels={
            "ClaimStartDt": "Claim Start Date",
            "ReimbursementINR": "Reimbursement (INR, converted from USD)",
            "SourceReimbursementUSD": "Source reimbursement (USD)",
        }
    )
    fig.update_traces(marker=dict(line=dict(width=1, color="DarkSlateGrey")))
    fig = format_inr_axis(fig, plot_df["ReimbursementINR"])
    return style_figure(fig, "Chronological Claim History & Reimbursement Outliers", height=380)


def plot_nfhs_screening_distribution(dist_dict: Dict[str, int]) -> go.Figure:
    """Bar chart for NFHS-5 medical consistency screening tiers."""
    order = ["Clean", "Low", "Medium", "High", "Critical"]
    counts = [dist_dict.get(k, 0) for k in order]
    colors = [THEME["success"], "#38BDF8", THEME["warning"], "#F97316", THEME["danger"]]

    fig = go.Figure(data=[go.Bar(
        x=order,
        y=counts,
        text=[f"{v:,}" for v in counts],
        textposition="auto",
        marker=dict(color=colors)
    )])
    fig.update_yaxes(type="log", title="Number of Respondents (Log Scale)", gridcolor="#F1F5F9")
    return style_figure(fig, "NFHS-5 Medical Consistency Screening Categories", height=340)


def plot_nfhs_rule_breakdown(rule_counts: Dict[str, int]) -> go.Figure:
    """Horizontal bar chart showing trigger counts across the 6 clinical consistency rules."""
    short_names = {
        "R1_diabetes_no_tx": "Diabetes w/o Treatment",
        "R1_htn_no_tx": "Hypertension w/o Treatment",
        "R2_tx_without_diabetes": "Treatment w/o Diabetes Dx",
        "R2_tx_without_htn": "Treatment w/o HTN Dx",
        "R3_high_glucose_no_dx": "High Glucose (>=200) w/o Dx",
        "R4_diabetes_no_meds": "Diabetes w/o Prescribed Meds"
    }

    labels = [short_names.get(k, k) for k in rule_counts.keys()]
    values = list(rule_counts.values())

    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        text=[f"{v:,}" for v in values],
        textposition="auto",
        marker_color=THEME["primary"]
    ))
    return style_figure(fig, "Clinical Screening Rules Triggered", height=340)
