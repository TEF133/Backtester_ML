"""
Commodity Volatility Dashboard
================================
Compares Implied vs Realized Volatility across commodity futures.

IV sources:
  ^OVX  — CBOE WTI Crude Oil Volatility Index   (CL=F)
  ^GVZ  — CBOE Gold Volatility Index              (GC=F)
  Others — current ATM IV from yfinance options chains (snapshot only)

Run:
    python evaluation/vol_dashboard.py
    → http://127.0.0.1:8050

Dependencies:
    pip install dash plotly yfinance pandas numpy scipy
"""

import sys
import time
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime

warnings.filterwarnings("ignore")

try:
    from dash import Dash, dcc, html, Input, Output, dash_table
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
except ImportError:
    print("Missing dependencies. Run:  pip install dash plotly")
    sys.exit(1)


# ── Universe ──────────────────────────────────────────────────────────────────
COMMODITIES = {
    "CL=F": {"name": "WTI Crude",    "iv_idx": "^OVX",  "color": "#e74c3c"},
    "GC=F": {"name": "Gold",         "iv_idx": "^GVZ",  "color": "#f1c40f"},
    "NG=F": {"name": "Natural Gas",  "iv_idx": None,    "color": "#5dade2"},
    "SI=F": {"name": "Silver",       "iv_idx": None,    "color": "#aab7b8"},
    "ZC=F": {"name": "Corn",         "iv_idx": None,    "color": "#58d68d"},
    "ZW=F": {"name": "Wheat",        "iv_idx": None,    "color": "#e67e22"},
    "HG=F": {"name": "Copper",       "iv_idx": None,    "color": "#a569bd"},
}

DEFAULT_START  = "2021-01-01"
RV_WINDOWS     = [10, 21, 63]          # realized-vol rolling windows (days)
CONE_WINDOWS   = [10, 21, 42, 63, 126] # vol-cone lookback windows
PERCENTILES    = [5, 25, 50, 75, 95]
ZSCORE_WINDOW  = 252                   # 1-year z-score normalisation

# ── Colour theme ──────────────────────────────────────────────────────────────
BG        = "#0d1117"
PANEL     = "#161b22"
BORDER    = "#30363d"
TXT       = "#c9d1d9"
TXT_DIM   = "#8b949e"
GREEN     = "#3fb950"
RED       = "#f85149"
YELLOW    = "#d29922"
GRID_CLR  = "#21262d"

PLOTLY_LAYOUT = dict(
    paper_bgcolor=BG,
    plot_bgcolor=PANEL,
    font_color=TXT,
    font_size=11,
    legend=dict(bgcolor=PANEL, bordercolor=BORDER, borderwidth=1),
    xaxis=dict(gridcolor=GRID_CLR, zerolinecolor=GRID_CLR, showspikes=True,
               spikethickness=1, spikecolor=TXT_DIM),
    yaxis=dict(gridcolor=GRID_CLR, zerolinecolor=GRID_CLR, showspikes=True,
               spikethickness=1, spikecolor=TXT_DIM),
    hovermode="x unified",
    margin=dict(l=50, r=20, t=40, b=40),
)


# ── In-memory cache ───────────────────────────────────────────────────────────
_cache: dict = {}
CACHE_TTL = 600   # seconds (10 min)


def _cached(key, fn, *args, **kwargs):
    if key in _cache:
        val, ts = _cache[key]
        if time.time() - ts < CACHE_TTL:
            return val
    val = fn(*args, **kwargs)
    _cache[key] = (val, time.time())
    return val


# ── Data helpers ──────────────────────────────────────────────────────────────

def _dl(tickers, start):
    """Download adjusted close prices; normalise to plain DataFrame."""
    raw = yf.download(tickers, start=start, auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
    else:
        close = raw[["Close"]].rename(columns={"Close": tickers[0]}) \
            if isinstance(tickers, list) and len(tickers) == 1 \
            else raw[["Close"]]
    return close.dropna(how="all")


def fetch_prices(start: str = DEFAULT_START) -> pd.DataFrame:
    tickers = list(COMMODITIES.keys())
    return _cached(f"prices_{start}", _dl, tickers, start)


def fetch_iv_series(iv_ticker: str, start: str = DEFAULT_START) -> pd.Series:
    raw = _cached(f"iv_{iv_ticker}_{start}", _dl, [iv_ticker], start)
    s = raw.squeeze()
    return (s / 100.0).rename(iv_ticker)


def compute_rv(close: pd.Series, windows=RV_WINDOWS) -> pd.DataFrame:
    """Annualised realised volatility at several rolling windows."""
    lr = np.log(close / close.shift(1))
    return pd.DataFrame(
        {f"rv{w}": lr.rolling(w).std() * np.sqrt(252) for w in windows},
        index=close.index,
    )


def vol_cone(close: pd.Series, windows=CONE_WINDOWS,
             pcts=PERCENTILES) -> tuple[pd.DataFrame, pd.Series]:
    """
    Returns:
        cone_df  — rows=window, cols=percentile levels (annualised)
        current  — current RV at each window
    """
    lr   = np.log(close / close.shift(1))
    rows, cur = [], []
    for w in windows:
        rv  = (lr.rolling(w).std() * np.sqrt(252)).dropna()
        row = {str(p): float(np.percentile(rv, p)) for p in pcts}
        row["window"] = w
        rows.append(row)
        cur.append(float(rv.iloc[-1]) if len(rv) else np.nan)
    return (pd.DataFrame(rows).set_index("window"),
            pd.Series(cur, index=windows, name="current"))


def fetch_term_structure(ticker: str) -> pd.DataFrame:
    """Extract ATM IV per expiry from yfinance options chains."""
    tk   = yf.Ticker(ticker)
    info = tk.fast_info
    spot = info.get("last_price") or info.get("regularMarketPrice")
    if spot is None:
        h    = tk.history(period="5d")
        spot = float(h["Close"].iloc[-1]) if not h.empty else None
    if spot is None or not tk.options:
        return pd.DataFrame()

    today   = datetime.today()
    records = []
    for exp_str in tk.options[:10]:
        try:
            chain  = tk.option_chain(exp_str)
            exp_dt = datetime.strptime(exp_str, "%Y-%m-%d")
            dte    = max((exp_dt - today).days, 1)

            calls = chain.calls.dropna(subset=["impliedVolatility"])
            puts  = chain.puts.dropna(subset=["impliedVolatility"])
            if calls.empty or puts.empty:
                continue
            c_atm = calls.iloc[(calls["strike"] - spot).abs().argmin()]
            p_atm = puts.iloc[(puts["strike"] - spot).abs().argmin()]
            records.append({
                "expiry":  exp_str,
                "dte":     dte,
                "call_iv": float(c_atm["impliedVolatility"]),
                "put_iv":  float(p_atm["impliedVolatility"]),
                "mid_iv":  (float(c_atm["impliedVolatility"]) +
                             float(p_atm["impliedVolatility"])) / 2,
                "strike":  float(c_atm["strike"]),
            })
        except Exception:
            continue
    return pd.DataFrame(records)


def snapshot_table(start: str = DEFAULT_START) -> pd.DataFrame:
    """Current-day summary metrics for all commodities."""
    prices = fetch_prices(start)
    rows   = []
    for tkr, meta in COMMODITIES.items():
        if tkr not in prices.columns:
            continue
        close = prices[tkr].dropna()
        if len(close) < 63:
            continue
        rv_df  = compute_rv(close, windows=[10, 21, 63])
        last   = close.iloc[-1]
        rv10   = rv_df["rv10"].iloc[-1]
        rv21   = rv_df["rv21"].iloc[-1]
        rv63   = rv_df["rv63"].iloc[-1]

        # IV: use index where available, else None
        iv_now = np.nan
        vrp    = np.nan
        vrp_z  = np.nan
        iv_idx = meta.get("iv_idx")
        if iv_idx:
            try:
                iv_s = fetch_iv_series(iv_idx, start)
                iv_s, rv21_s = iv_s.align(rv_df["rv21"], join="inner")
                iv_now = float(iv_s.iloc[-1])
                spread = (iv_s - rv21_s).dropna()
                vrp    = float(spread.iloc[-1])
                vrp_z  = float((spread - spread.rolling(ZSCORE_WINDOW).mean()).iloc[-1] /
                               (spread.rolling(ZSCORE_WINDOW).std().iloc[-1] + 1e-8))
            except Exception:
                pass

        rows.append({
            "Ticker":   tkr,
            "Name":     meta["name"],
            "Price":    round(last, 2),
            "RV 10d":   f"{rv10:.1%}",
            "RV 21d":   f"{rv21:.1%}",
            "RV 63d":   f"{rv63:.1%}",
            "IV (idx)": f"{iv_now:.1%}" if not np.isnan(iv_now) else "—",
            "IV-RV":    f"{vrp:+.1%}" if not np.isnan(vrp) else "-",
            "VRP Z":    f"{vrp_z:+.2f}" if not np.isnan(vrp_z) else "—",
            "_vrp_z":   vrp_z,   # numeric for colouring
        })
    return pd.DataFrame(rows)


# ── Figure builders ───────────────────────────────────────────────────────────

def fig_iv_rv(ticker: str, start: str) -> go.Figure:
    """3-panel: price | IV vs RV | VRP z-score."""
    meta  = COMMODITIES[ticker]
    close = fetch_prices(start)[ticker].dropna()
    rv_df = compute_rv(close)

    iv_idx = meta.get("iv_idx")
    has_iv = iv_idx is not None
    iv_s   = None
    if has_iv:
        try:
            iv_s = fetch_iv_series(iv_idx, start)
        except Exception:
            has_iv = False

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.35, 0.40, 0.25],
        vertical_spacing=0.03,
        subplot_titles=[
            f"{meta['name']} — Price",
            "Implied vs Realised Volatility",
            "Vol Risk Premium  Z-score  (IV − RV21d)",
        ],
    )

    # Panel 1 — price
    fig.add_trace(go.Scatter(
        x=close.index, y=close.values,
        mode="lines", name="Price",
        line=dict(color=meta["color"], width=1.5),
        hovertemplate="%{y:.2f}",
    ), row=1, col=1)

    # Panel 2 — RV lines
    rv_styles = [("rv10", "RV 10d", TXT_DIM, "dot"),
                 ("rv21", "RV 21d", "#58d68d", "solid"),
                 ("rv63", "RV 63d", "#5dade2", "dash")]
    for col, label, clr, dash in rv_styles:
        fig.add_trace(go.Scatter(
            x=rv_df.index, y=(rv_df[col] * 100).values,
            mode="lines", name=label,
            line=dict(color=clr, width=1.4, dash=dash),
            hovertemplate="%{y:.1f}%",
        ), row=2, col=1)

    if has_iv and iv_s is not None:
        fig.add_trace(go.Scatter(
            x=iv_s.index, y=(iv_s * 100).values,
            mode="lines", name=f"IV ({iv_idx})",
            line=dict(color=RED, width=2),
            hovertemplate="%{y:.1f}%",
        ), row=2, col=1)

        # Panel 3 — VRP z-score
        iv_a, rv_a = iv_s.align(rv_df["rv21"], join="inner")
        spread = (iv_a - rv_a).dropna()
        vrp_z  = (spread - spread.rolling(ZSCORE_WINDOW).mean()) / \
                 (spread.rolling(ZSCORE_WINDOW).std() + 1e-8)
        colors = [GREEN if v < 0 else RED for v in vrp_z.values]
        fig.add_trace(go.Bar(
            x=vrp_z.index, y=vrp_z.values,
            name="VRP Z", marker_color=colors,
            hovertemplate="%{y:.2f}",
        ), row=3, col=1)
        fig.add_hline(y=1.0,  line_dash="dash", line_color=RED,
                      line_width=1, row=3, col=1)
        fig.add_hline(y=-1.0, line_dash="dash", line_color=GREEN,
                      line_width=1, row=3, col=1)
    else:
        fig.add_annotation(
            text="No IV index available — showing RV only",
            xref="x domain", yref="y domain",
            x=0.5, y=0.5, row=3, col=1,
            showarrow=False, font=dict(color=TXT_DIM, size=12),
        )

    fig.update_layout(**PLOTLY_LAYOUT, height=680,
                      title_text=f"Vol Analysis — {meta['name']}")
    fig.update_yaxes(row=2, ticksuffix="%")
    return fig


def fig_vol_cone(ticker: str, start: str) -> go.Figure:
    """Vol cone: RV percentile bands across horizons + current."""
    meta  = COMMODITIES[ticker]
    close = fetch_prices(start)[ticker].dropna()
    cone, cur = vol_cone(close)

    fig  = go.Figure()
    band_pairs = list(zip(PERCENTILES, PERCENTILES[::-1]))[:len(PERCENTILES)//2]
    alphas     = [0.15, 0.25, 0.35]

    for i, (lo, hi) in enumerate([(5, 95), (25, 75)]):
        rgba = f"rgba(93,173,226,{alphas[i]})"
        fig.add_trace(go.Scatter(
            x=CONE_WINDOWS + CONE_WINDOWS[::-1],
            y=(cone[str(lo)].tolist() + cone[str(hi)].tolist()[::-1]),
            fill="toself", fillcolor=rgba,
            line=dict(color="rgba(0,0,0,0)"),
            name=f"{lo}–{hi} pct",
            hoverinfo="skip",
        ))

    fig.add_trace(go.Scatter(
        x=CONE_WINDOWS, y=(cone["50"] * 100).tolist(),
        mode="lines+markers", name="Median (50th)",
        line=dict(color=TXT_DIM, dash="dash", width=1),
        marker=dict(size=5),
        hovertemplate="%{y:.1f}%",
    ))
    fig.add_trace(go.Scatter(
        x=CONE_WINDOWS, y=(cur * 100).tolist(),
        mode="lines+markers", name="Current RV",
        line=dict(color=meta["color"], width=2.5),
        marker=dict(size=8, symbol="diamond"),
        hovertemplate="%{y:.1f}%",
    ))

    layout = {**PLOTLY_LAYOUT,
              "height": 450,
              "title_text": f"Realised Vol Cone — {meta['name']}",
              "xaxis_title": "Lookback window (trading days)",
              "yaxis_title": "Annualised Vol (%)",
              "yaxis": dict(**PLOTLY_LAYOUT["yaxis"], ticksuffix="%")}
    fig.update_layout(**layout)
    fig.update_xaxes(tickvals=CONE_WINDOWS,
                     ticktext=[f"{w}d" for w in CONE_WINDOWS])
    return fig


def fig_term_structure(ticker: str) -> go.Figure:
    """ATM IV across option expiries (current snapshot)."""
    meta = COMMODITIES[ticker]
    ts   = _cached(f"ts_{ticker}", fetch_term_structure, ticker)

    fig = go.Figure()
    if ts.empty:
        fig.add_annotation(
            text="No options data available for this ticker",
            x=0.5, y=0.5, xref="paper", yref="paper",
            showarrow=False, font=dict(color=TXT_DIM, size=14),
        )
    else:
        fig.add_trace(go.Scatter(
            x=ts["dte"], y=(ts["call_iv"] * 100),
            mode="lines+markers", name="Call ATM IV",
            line=dict(color=GREEN, width=2),
            marker=dict(size=7),
            hovertemplate="DTE %{x}d → %{y:.1f}%",
        ))
        fig.add_trace(go.Scatter(
            x=ts["dte"], y=(ts["put_iv"] * 100),
            mode="lines+markers", name="Put ATM IV",
            line=dict(color=RED, width=2),
            marker=dict(size=7),
            hovertemplate="DTE %{x}d → %{y:.1f}%",
        ))
        fig.add_trace(go.Scatter(
            x=ts["dte"], y=(ts["mid_iv"] * 100),
            mode="lines+markers", name="Mid IV",
            line=dict(color=meta["color"], width=2.5, dash="dot"),
            marker=dict(size=8),
            hovertemplate="DTE %{x}d → %{y:.1f}%",
        ))
        for _, row in ts.iterrows():
            fig.add_annotation(
                x=row["dte"], y=row["mid_iv"] * 100,
                text=f"  {row['expiry'][5:]}",   # MM-DD
                showarrow=False, xanchor="left",
                font=dict(size=9, color=TXT_DIM),
            )

    layout = {**PLOTLY_LAYOUT,
              "height": 420,
              "title_text": f"IV Term Structure — {meta['name']} (current snapshot)",
              "xaxis_title": "Days to expiry",
              "yaxis_title": "ATM Implied Vol (%)"}
    layout["yaxis"] = dict(**PLOTLY_LAYOUT["yaxis"], ticksuffix="%")
    fig.update_layout(**layout)
    return fig


def fig_snapshot_heatmap(df: pd.DataFrame) -> go.Figure:
    """Heatmap of VRP z-scores across commodities."""
    if df.empty:
        return go.Figure()

    # Only rows with numeric VRP z
    d = df.dropna(subset=["_vrp_z"])
    if d.empty:
        return go.Figure(layout=go.Layout(
            annotations=[dict(text="No IV index data available",
                              x=0.5, y=0.5, xref="paper", yref="paper",
                              showarrow=False, font=dict(color=TXT_DIM, size=14))],
            **PLOTLY_LAYOUT,
        ))

    fig = go.Figure(go.Bar(
        x=d["Name"],
        y=d["_vrp_z"],
        marker_color=[RED if v > 0 else GREEN for v in d["_vrp_z"]],
        text=[f"{v:+.2f}" for v in d["_vrp_z"]],
        textposition="outside",
        hovertemplate="%{x}<br>VRP Z = %{y:.2f}<extra></extra>",
    ))
    fig.add_hline(y=1.0,  line_dash="dash", line_color=RED,   line_width=1,
                  annotation_text=" IV rich (+1σ)", annotation_font_color=RED)
    fig.add_hline(y=-1.0, line_dash="dash", line_color=GREEN, line_width=1,
                  annotation_text=" IV cheap (−1σ)", annotation_font_color=GREEN)
    fig.add_hline(y=0, line_color=BORDER, line_width=1)
    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=340,
        title_text="Vol Risk Premium Z-score — Cross-Commodity Snapshot",
        yaxis_title="Z-score (IV − RV21d)",
        showlegend=False,
    )
    return fig


# ── Dash layout ───────────────────────────────────────────────────────────────

TICKER_OPTIONS = [
    {"label": f"{v['name']} ({k})", "value": k}
    for k, v in COMMODITIES.items()
]

app = Dash(__name__, title="Commodity Vol Dashboard")
app.layout = html.Div(
    style={"backgroundColor": BG, "color": TXT,
           "fontFamily": "'Segoe UI', Arial, sans-serif", "minHeight": "100vh"},
    children=[

        # ── Header ──────────────────────────────────────────────────────────
        html.Div(
            style={"backgroundColor": PANEL, "borderBottom": f"1px solid {BORDER}",
                   "padding": "14px 24px", "display": "flex",
                   "alignItems": "center", "gap": "24px"},
            children=[
                html.H2("Commodity Volatility Dashboard",
                        style={"margin": 0, "fontSize": "18px",
                               "color": TXT, "fontWeight": 600}),
                html.Span("IV vs RV | Vol Cone | Term Structure",
                          style={"color": TXT_DIM, "fontSize": "12px"}),
            ],
        ),

        # ── Controls ─────────────────────────────────────────────────────────
        html.Div(
            style={"backgroundColor": PANEL, "borderBottom": f"1px solid {BORDER}",
                   "padding": "10px 24px", "display": "flex",
                   "alignItems": "center", "gap": "32px", "flexWrap": "wrap"},
            children=[
                html.Div([
                    html.Label("Commodity", style={"fontSize": "11px",
                               "color": TXT_DIM, "marginBottom": "4px",
                               "display": "block"}),
                    dcc.Dropdown(
                        id="ticker-select",
                        options=TICKER_OPTIONS,
                        value="CL=F",
                        clearable=False,
                        style={"width": "210px", "fontSize": "13px",
                               "backgroundColor": BG, "color": TXT,
                               "borderColor": BORDER},
                    ),
                ]),
                html.Div([
                    html.Label("History start", style={"fontSize": "11px",
                               "color": TXT_DIM, "marginBottom": "4px",
                               "display": "block"}),
                    dcc.Dropdown(
                        id="start-select",
                        options=[
                            {"label": "1 year",  "value": "2024-01-01"},
                            {"label": "2 years", "value": "2023-01-01"},
                            {"label": "3 years", "value": "2022-01-01"},
                            {"label": "5 years", "value": "2020-01-01"},
                        ],
                        value=DEFAULT_START,
                        clearable=False,
                        style={"width": "130px", "fontSize": "13px",
                               "backgroundColor": BG, "color": TXT,
                               "borderColor": BORDER},
                    ),
                ]),
                html.Div([
                    html.Label("IV note", style={"fontSize": "11px",
                               "color": TXT_DIM, "display": "block",
                               "marginBottom": "4px"}),
                    html.Span(
                        "Historical IV: ^OVX (crude) | ^GVZ (gold) | others = RV only",
                        style={"fontSize": "11px", "color": TXT_DIM},
                    ),
                ]),
            ],
        ),

        # ── Tabs ─────────────────────────────────────────────────────────────
        html.Div(style={"padding": "0 24px"}, children=[
            dcc.Tabs(
                id="tabs",
                value="tab-ivrv",
                style={"borderBottom": f"1px solid {BORDER}"},
                colors={"border": BORDER, "primary": "#58a6ff",
                        "background": PANEL},
                children=[
                    dcc.Tab(label="IV vs RV",        value="tab-ivrv"),
                    dcc.Tab(label="Vol Cone",         value="tab-cone"),
                    dcc.Tab(label="Term Structure",   value="tab-ts"),
                    dcc.Tab(label="Snapshot",         value="tab-snap"),
                ],
            ),
            dcc.Loading(
                type="dot",
                color="#58a6ff",
                children=html.Div(id="tab-content",
                                  style={"paddingTop": "12px"}),
            ),
        ]),
    ],
)


# ── Callbacks ─────────────────────────────────────────────────────────────────

@app.callback(
    Output("tab-content", "children"),
    Input("tabs", "value"),
    Input("ticker-select", "value"),
    Input("start-select", "value"),
)
def render_tab(tab: str, ticker: str, start: str):
    if tab == "tab-ivrv":
        return dcc.Graph(figure=fig_iv_rv(ticker, start),
                         config={"displayModeBar": True,
                                 "scrollZoom": True})

    if tab == "tab-cone":
        return dcc.Graph(figure=fig_vol_cone(ticker, start),
                         config={"displayModeBar": True})

    if tab == "tab-ts":
        return dcc.Graph(figure=fig_term_structure(ticker),
                         config={"displayModeBar": True})

    if tab == "tab-snap":
        snap = snapshot_table(start)
        vrp_fig = fig_snapshot_heatmap(snap)
        display_cols = [c for c in snap.columns if not c.startswith("_")]
        table = dash_table.DataTable(
            data=snap[display_cols].to_dict("records"),
            columns=[{"name": c, "id": c} for c in display_cols],
            style_table={"overflowX": "auto"},
            style_cell={
                "backgroundColor": PANEL, "color": TXT,
                "border": f"1px solid {BORDER}",
                "fontFamily": "'Segoe UI', Arial, sans-serif",
                "fontSize": "13px", "padding": "8px 12px",
                "textAlign": "right",
            },
            style_header={
                "backgroundColor": ACCENT if False else "#21262d",
                "color": TXT, "fontWeight": "600",
                "borderBottom": f"2px solid {BORDER}",
            },
            style_data_conditional=[
                {"if": {"filter_query": '{VRP Z} contains "+"',
                        "column_id": "VRP Z"},
                 "color": RED},
                {"if": {"filter_query": '{VRP Z} contains "-"',
                        "column_id": "VRP Z"},
                 "color": GREEN},
                {"if": {"column_id": "Name"},
                 "textAlign": "left"},
            ],
        )
        return html.Div([
            dcc.Graph(figure=vrp_fig, config={"displayModeBar": False}),
            html.Div(style={"marginTop": "16px"}, children=[table]),
        ])

    return html.Div("Select a tab.")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\nCommodity Vol Dashboard")
    print("  → http://127.0.0.1:8050\n")
    print("  Tab 1 — IV vs RV:      historical IV index vs rolling RV, VRP z-score")
    print("  Tab 2 — Vol Cone:      RV percentile bands across horizons")
    print("  Tab 3 — Term Struct:   current ATM IV across option expiries")
    print("  Tab 4 — Snapshot:      cross-commodity VRP summary table\n")
    app.run(debug=False, host="127.0.0.1", port=8050)
