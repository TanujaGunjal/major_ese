"""
Enterprise styling module for the Healthcare Fraud Investigation Dashboard.
Provides clean healthcare analytics aesthetics, subtle cards, and high-contrast badges
without conflicting with Streamlit's native input elements.
"""

import streamlit as st


def apply_custom_styles():
    """Injects safe, clean CSS that does not break input widgets or text contrast."""
    st.markdown("""
    <style>
    :root {
        --dashboard-ink: #0F172A;
        --dashboard-muted: #475569;
        --dashboard-primary: #0D5C75;
        --dashboard-border: #E2E8F0;
        --dashboard-surface: #FFFFFF;
    }

    /* Keep the native navigation legible in both light- and dark-mode browsers. */
    section[data-testid="stSidebar"],
    section[data-testid="stSidebar"] > div,
    section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
        background: #FFFFFF !important;
    }
    section[data-testid="stSidebar"] *,
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] span {
        color: var(--dashboard-ink);
    }
    section[data-testid="stSidebar"] [data-testid="stSidebarNavLink"] {
        color: var(--dashboard-ink) !important;
        background-color: #FFFFFF !important;
        border: 1px solid transparent;
        border-radius: 9px;
        margin: 3px 8px;
        min-height: 42px;
    }
    section[data-testid="stSidebar"] [data-testid="stSidebarNavLink"] p,
    section[data-testid="stSidebar"] [data-testid="stSidebarNavLink"] span {
        color: var(--dashboard-ink) !important;
        font-weight: 600 !important;
        opacity: 1 !important;
    }
    section[data-testid="stSidebar"] [data-testid="stSidebarNavLink"]:hover {
        background-color: #EFF6FF !important;
        border-color: #BFDBFE !important;
    }
    section[data-testid="stSidebar"] [data-testid="stSidebarNavLink"][aria-current="page"] {
        background-color: #E0F2FE !important;
        border-color: #7DD3FC !important;
        box-shadow: inset 3px 0 0 #0284C7;
    }
    section[data-testid="stSidebar"] [data-testid="stSidebarNavLink"][aria-current="page"] p,
    section[data-testid="stSidebar"] [data-testid="stSidebarNavLink"][aria-current="page"] span {
        color: #0C4A6E !important;
        font-weight: 700 !important;
    }
    section[data-testid="stSidebar"] [data-testid="stSidebarNavLink"] svg {
        fill: currentColor !important;
    }
    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
        color: var(--dashboard-muted) !important;
    }
    section[data-testid="stSidebar"] [data-testid="stButton"] button {
        color: var(--dashboard-ink) !important;
    }

    /* Keep the browser-facing toolbar limited to useful presentation controls. */
    header [data-testid="stBaseButton-header"] {
        display: none !important;
    }
    [data-testid="stMainMenuItem-rerun"],
    [data-testid="stMainMenuItem-autoRerun"],
    [data-testid="stMainMenuItem-clearCache"],
    [data-testid="stMainMenuItem-recordScreencast"],
    [data-testid="stMainMenuDivider"] {
        display: none !important;
    }
    [data-testid="stMainMenuPopover"] > div > div:nth-child(2),
    footer {
        display: none !important;
    }

    /* Consistent page typography and surfaces. */
    .stApp, [data-testid="stMain"] {
        color: var(--dashboard-ink);
    }
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stMarkdownContainer"] span {
        line-height: 1.55;
    }
    [data-testid="stDataFrame"] {
        border: 1px solid var(--dashboard-border);
        border-radius: 10px;
        overflow: hidden;
    }

    /* Metric Card Polish */
    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        padding: 16px 20px;
        border-radius: 10px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.04);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.08);
        border-color: #CBD5E1;
    }
    div[data-testid="stMetricLabel"] {
        color: #475569 !important;
        font-weight: 600 !important;
        font-size: 13px !important;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }
    div[data-testid="stMetricValue"] {
        color: #0F172A !important;
        font-weight: 700 !important;
        font-size: 26px !important;
    }

    /* Clean Hero Header */
    .hero-banner {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        padding: 24px 30px;
        border-radius: 12px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .hero-banner h1 {
        color: #FFFFFF !important;
        font-size: 26px !important;
        font-weight: 700 !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    .hero-banner p {
        color: #94A3B8 !important;
        font-size: 14px !important;
        margin: 6px 0 0 0 !important;
        padding: 0 !important;
    }

    /* Risk Badges */
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 12px;
        font-weight: 600;
        margin-right: 6px;
    }
    .badge-high {
        background-color: #FEE2E2;
        color: #DC2626;
        border: 1px solid #FCA5A5;
    }
    .badge-medium {
        background-color: #FEF3C7;
        color: #D97706;
        border: 1px solid #FCD34D;
    }
    .badge-low {
        background-color: #DCFCE7;
        color: #16A34A;
        border: 1px solid #86EFAC;
    }
    </style>
    """, unsafe_allow_html=True)


def render_header(title: str, subtitle: str):
    """Renders a standard high-contrast hero header."""
    st.markdown(f"""
    <div class="hero-banner">
        <h1>{title}</h1>
        <p>{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)


def render_metric_card(title: str, value: str, subtitle: str = ""):
    """Renders a native styled metric card."""
    st.metric(label=title, value=value, help=subtitle if subtitle else None)
