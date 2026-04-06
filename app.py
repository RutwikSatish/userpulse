"""
UserPulse — User Engagement Scoring, Churn Risk Detection & Activation Analytics
Resume claims: 5,000+ users · 50 accounts · 4 behavioral dimensions
               Power/Active/At-Risk/Dormant segmentation · ranked outreach engine
               cohort retention dashboard · ROI estimation per action
Tools: Python · pandas · scikit-learn · Streamlit · Plotly · SQLite
"""

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, timedelta
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="UserPulse · Engagement Analytics",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding-top:1.4rem;padding-bottom:2rem;}
[data-testid="metric-container"]{
  background:#0D1221;border:1px solid #1C2640;border-radius:10px;padding:1rem 1.2rem;}
[data-testid="metric-container"] label{
  font-size:11px!important;text-transform:uppercase;letter-spacing:.06em;color:#6B7FA8!important;}
[data-testid="metric-container"] [data-testid="stMetricValue"]{
  font-size:1.85rem!important;font-weight:800!important;}
.stTabs [data-baseweb="tab-list"]{gap:0;border-bottom:1px solid #1C2640;background:transparent;}
.stTabs [data-baseweb="tab"]{padding:.65rem 1.2rem;font-size:13px;font-weight:600;color:#6B7FA8;border-bottom:2px solid transparent;}
.stTabs [aria-selected="true"]{color:#EEF0FF!important;border-bottom:2px solid #6366F1!important;background:transparent!important;}
.stTabs [data-baseweb="tab-highlight"]{display:none;}
.stTabs [data-baseweb="tab-border"]{display:none;}
[data-testid="stSidebar"]{background:#090E1A;border-right:1px solid #1C2640;}
div[data-testid="stDataFrame"] {border-radius:10px;overflow:hidden;}
</style>
""", unsafe_allow_html=True)

INDIGO = "#6366F1"
GREEN  = "#10B981"
AMBER  = "#F59E0B"
RED    = "#EF4444"
GRAY   = "#6B7FA8"

PLOT = dict(
    plot_bgcolor="#0D1221", paper_bgcolor="#0D1221",
    font=dict(color="#6B7FA8", family="Inter"),
    margin=dict(l=0, r=0, t=20, b=0),
)

TIER_COLORS = {"Power": GREEN, "Active": INDIGO, "At-Risk": AMBER, "Dormant": RED}

INDUSTRIES = ["Financial Services", "Technology", "Healthcare", "Legal", "Consulting",
              "Private Equity", "Asset Management", "Media", "Pharma", "Government"]

ACCOUNT_NAMES = [
    "Apex Capital", "BrightPath Solutions", "Crestline Partners", "Delta Advisory",
    "Everest Group", "Fortis Ventures", "Granite Consulting", "Harbor Analytics",
    "Ironclad Management", "Jasper Law Group", "Keystone Research", "Luminary Capital",
    "Meridian Health", "Nexus Partners", "Onyx Financial", "Pinnacle Law",
    "Quantum Analytics", "Redwood Advisors", "Summit Equity", "Titan Consulting",
    "Unison Capital", "Valor Group", "Westfield Research", "Axiom Partners",
    "Blueprint Ventures", "Cascade Capital", "Dover Analytics", "Ember Group",
    "Flagship Capital", "Greenvale Partners", "Horizon Law", "Invictus Consulting",
    "Juniper Research", "Kestrel Capital", "Lodestar Analytics", "Maple Partners",
    "Northstar Equity", "Osprey Ventures", "Paragon Research", "Quorum Capital",
    "Raven Consulting", "Solaris Partners", "Tempest Analytics", "Uplift Capital",
    "Vantage Research", "Wavecrest Partners", "Xenon Capital", "Yellowstone Law",
    "Zenith Consulting", "Azimuth Group",
]

ENGAGEMENT_TYPES = {
    "Power":   {"action": "Expansion outreach — propose seat increase",  "roi": (800, 2000)},
    "Active":  {"action": "Feature adoption nudge — schedule demo",       "roi": (200, 600)},
    "At-Risk": {"action": "Re-engagement call — success team check-in",   "roi": (150, 400)},
    "Dormant": {"action": "Reactivation campaign — personalised email",   "roi": (50,  200)},
}


@st.cache_data
def generate_data(seed=42):
    rng = np.random.default_rng(seed)
    today = date.today()
    n_accounts = 50
    users_per_account = np.round(rng.normal(100, 25, n_accounts)).clip(40, 180).astype(int)

    records = []
    for acct_idx in range(n_accounts):
        name     = ACCOUNT_NAMES[acct_idx]
        industry = INDUSTRIES[acct_idx % len(INDUSTRIES)]
        acct_val = int(rng.integers(15_000, 200_000))
        renewal  = today + timedelta(days=int(rng.integers(14, 365)))
        n_users  = users_per_account[acct_idx]

        for u in range(n_users):
            # Behavioral signals (0-100 scale)
            login_freq      = float(np.clip(rng.normal(50, 30), 0, 100))
            session_depth   = float(np.clip(rng.normal(50, 28), 0, 100))
            feature_adoption= float(np.clip(rng.normal(45, 32), 0, 100))
            query_volume    = float(np.clip(rng.normal(48, 30), 0, 100))

            # Weighted engagement score
            eng_score = (
                0.30 * login_freq +
                0.25 * session_depth +
                0.25 * feature_adoption +
                0.20 * query_volume
            )

            # Tier
            if eng_score >= 72:   tier = "Power"
            elif eng_score >= 46: tier = "Active"
            elif eng_score >= 22: tier = "At-Risk"
            else:                 tier = "Dormant"

            # Days since last active (inversely related to engagement)
            base_days = max(0, 90 - eng_score + rng.normal(0, 8))
            days_since = int(np.clip(base_days, 0, 180))
            last_active = today - timedelta(days=days_since)

            # Churn probability (logistic-style)
            churn_logit = -3.5 + 0.055*(100 - eng_score) + 0.012*days_since
            churn_prob  = round(1 / (1 + np.exp(-churn_logit)), 3)

            # Estimated ROI
            lo, hi = ENGAGEMENT_TYPES[tier]["roi"]
            est_roi = int(rng.integers(lo, hi))

            records.append({
                "user_id":          f"U{acct_idx:02d}{u:03d}",
                "account":          name,
                "industry":         industry,
                "account_value":    acct_val,
                "renewal_date":     renewal.isoformat(),
                "days_to_renewal":  (renewal - today).days,
                "login_freq":       round(login_freq, 1),
                "session_depth":    round(session_depth, 1),
                "feature_adoption": round(feature_adoption, 1),
                "query_volume":     round(query_volume, 1),
                "engagement_score": round(eng_score, 1),
                "tier":             tier,
                "days_since_active":days_since,
                "last_active":      last_active.isoformat(),
                "churn_prob":       churn_prob,
                "recommended_action": ENGAGEMENT_TYPES[tier]["action"],
                "est_roi":          est_roi,
                "week_cohort":      (today - last_active).days // 7,
            })

    df = pd.DataFrame(records)
    return df


@st.cache_resource
def build_db(df):
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    df.to_sql("users", conn, index=False, if_exists="replace")
    return conn


# ── Load ────────────────────────────────────────────────────
with st.spinner("Loading 5,000+ user records..."):
    df_all = generate_data()
    conn   = build_db(df_all)

total_users    = len(df_all)
total_accounts = df_all["account"].nunique()
tier_counts    = df_all["tier"].value_counts()
churn_risk_pct = round((df_all["churn_prob"] > 0.5).mean() * 100, 1)
at_risk_dormant= tier_counts.get("At-Risk", 0) + tier_counts.get("Dormant", 0)


# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">
      <div style="width:34px;height:34px;border-radius:9px;
           background:linear-gradient(135deg,#6366F1,#8B5CF6);
           display:flex;align-items:center;justify-content:center;font-size:18px;">📡</div>
      <div>
        <div style="font-size:16px;font-weight:800;color:#EEF0FF;line-height:1.1;">UserPulse</div>
        <div style="font-size:10px;color:#6B7FA8;text-transform:uppercase;letter-spacing:.05em;">Engagement Analytics</div>
      </div>
    </div>""", unsafe_allow_html=True)

    st.markdown(
        '<span style="background:rgba(99,102,241,.12);border:1px solid rgba(99,102,241,.3);'
        'color:#818CF8;border-radius:20px;padding:3px 12px;font-size:11px;font-weight:700;">'
        f'{total_users:,} USERS · {total_accounts} ACCOUNTS</span>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<p style="font-size:12px;font-weight:700;color:#EEF0FF;margin-bottom:6px;">Filters</p>',
                unsafe_allow_html=True)
    sel_tiers = st.multiselect("Tier", ["Power","Active","At-Risk","Dormant"],
                               default=["Power","Active","At-Risk","Dormant"],
                               label_visibility="collapsed")
    sel_industry = st.multiselect("Industry", sorted(df_all["industry"].unique()),
                                  default=sorted(df_all["industry"].unique()),
                                  label_visibility="collapsed")
    st.markdown("---")
    st.markdown(f'<p style="font-size:11px;color:#3B4D63;">Dataset: {total_users:,} users · '
                f'{total_accounts} accounts · 4 behavioral dimensions · Fixed seed</p>',
                unsafe_allow_html=True)

df = df_all[df_all["tier"].isin(sel_tiers) & df_all["industry"].isin(sel_industry)].copy()


# ── TABS ────────────────────────────────────────────────────
t1, t2, t3, t4 = st.tabs([
    "  📊  Overview  ",
    "  🎯  User Segments  ",
    "  📬  Outreach Engine  ",
    "  📈  Cohort Analytics  ",
])


# ════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ════════════════════════════════════════════════════════════
with t1:
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Users",       f"{len(df):,}")
    m2.metric("Power Users",       f"{(df['tier']=='Power').sum():,}",
              help="Engagement score >= 72")
    m3.metric("At-Risk + Dormant", f"{((df['tier'].isin(['At-Risk','Dormant']))).sum():,}")
    m4.metric("Churn Risk > 50%",  f"{(df['churn_prob']>0.5).sum():,}")
    m5.metric("Avg Engagement Score", f"{df['engagement_score'].mean():.1f}")

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns([1, 1])

    with c1:
        st.markdown('<p style="font-size:11px;color:#6B7FA8;text-transform:uppercase;'
                    'letter-spacing:.06em;font-weight:600;">User tier distribution</p>',
                    unsafe_allow_html=True)
        tc = df["tier"].value_counts().reset_index()
        tc.columns = ["tier", "count"]
        fig1 = go.Figure(go.Pie(
            labels=tc["tier"], values=tc["count"], hole=0.54,
            marker_colors=[TIER_COLORS.get(t, GRAY) for t in tc["tier"]],
            textinfo="label+percent", textfont=dict(size=11, color="#EEF0FF"),
        ))
        fig1.update_layout(**PLOT, height=210, showlegend=False)
        st.plotly_chart(fig1, use_container_width=True, config={"displayModeBar": False})

    with c2:
        st.markdown('<p style="font-size:11px;color:#6B7FA8;text-transform:uppercase;'
                    'letter-spacing:.06em;font-weight:600;">Avg engagement score by industry</p>',
                    unsafe_allow_html=True)
        ind_eng = df.groupby("industry")["engagement_score"].mean().reset_index().sort_values("engagement_score")
        fig2 = go.Figure(go.Bar(
            x=ind_eng["engagement_score"], y=ind_eng["industry"], orientation="h",
            marker_color=[INDIGO if v >= df["engagement_score"].mean() else AMBER
                          for v in ind_eng["engagement_score"]],
            text=[f"{v:.1f}" for v in ind_eng["engagement_score"]],
            textposition="outside", textfont=dict(size=9, color="#6B7FA8"),
        ))
        fig2.update_layout(**PLOT, height=210,
            xaxis=dict(showgrid=False, showticklabels=False),
            yaxis=dict(showgrid=False, tickfont=dict(color="#94A3B8", size=10)))
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    # Account health heatmap
    st.markdown('<p style="font-size:11px;color:#6B7FA8;text-transform:uppercase;'
                'letter-spacing:.06em;font-weight:600;margin-top:4px;">'
                'Account health heatmap — % power users vs churn risk</p>',
                unsafe_allow_html=True)
    acct_health = df.groupby("account").agg(
        power_pct=("tier", lambda x: (x=="Power").mean()*100),
        churn_avg=("churn_prob", "mean"),
        users=("user_id", "count"),
        acct_val=("account_value", "first"),
    ).reset_index()
    fig3 = go.Figure(go.Scatter(
        x=acct_health["power_pct"],
        y=acct_health["churn_avg"]*100,
        mode="markers",
        marker=dict(
            size=acct_health["acct_val"]/8000 + 6,
            color=acct_health["power_pct"],
            colorscale=[[0, RED],[0.4, AMBER],[1.0, GREEN]],
            showscale=True,
            colorbar=dict(title="Power %", tickfont=dict(color="#6B7FA8", size=9), thickness=8),
        ),
        text=acct_health["account"],
        hovertemplate="<b>%{text}</b><br>Power users: %{x:.1f}%<br>Avg churn risk: %{y:.1f}%<extra></extra>",
    ))
    fig3.update_layout(**PLOT, height=230,
        xaxis=dict(showgrid=False, title="% Power Users", tickfont=dict(color="#94A3B8")),
        yaxis=dict(showgrid=False, title="Avg Churn Risk %", tickfont=dict(color="#94A3B8")),
    )
    fig3.add_hline(y=50, line_dash="dot", line_color=AMBER, line_width=1,
                   annotation_text="50% churn threshold", annotation_font_color=AMBER,
                   annotation_font_size=10)
    st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})


# ════════════════════════════════════════════════════════════
# TAB 2 — USER SEGMENTS
# ════════════════════════════════════════════════════════════
with t2:
    st.markdown("##### User Segmentation — Behavioral Dimension Profiles")

    s1, s2, s3, s4 = st.columns(4)
    for col, tier in zip([s1, s2, s3, s4], ["Power","Active","At-Risk","Dormant"]):
        sub = df[df["tier"]==tier]
        col.metric(
            f"{tier} users",
            f"{len(sub):,}",
            f"{len(sub)/len(df)*100:.1f}% of portfolio"
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Radar chart per tier
    st.markdown('<p style="font-size:11px;color:#6B7FA8;text-transform:uppercase;'
                'letter-spacing:.06em;font-weight:600;">Behavioral dimension radar by tier</p>',
                unsafe_allow_html=True)
    dims = ["login_freq","session_depth","feature_adoption","query_volume"]
    dim_labels = ["Login Freq","Session Depth","Feature Adoption","Query Volume"]

    fig4 = go.Figure()
    for tier, color in TIER_COLORS.items():
        sub = df[df["tier"]==tier]
        if len(sub) == 0:
            continue
        vals = [sub[d].mean() for d in dims]
        vals_closed = vals + [vals[0]]
        labels_closed = dim_labels + [dim_labels[0]]
        fig4.add_trace(go.Scatterpolar(
            r=vals_closed, theta=labels_closed, fill="toself",
            name=tier, line=dict(color=color, width=2),
            fillcolor=color.replace("#", "rgba(") + ",0.08)" if "#" in color else color,
            opacity=0.85,
        ))
    fig4.update_layout(**PLOT, height=280,
        polar=dict(
            bgcolor="#0D1221",
            radialaxis=dict(visible=True, range=[0,100], tickfont=dict(color="#6B7FA8", size=9),
                           gridcolor="#1C2640"),
            angularaxis=dict(tickfont=dict(color="#94A3B8", size=11), gridcolor="#1C2640"),
        ),
        legend=dict(font=dict(size=11, color="#94A3B8"), bgcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig4, use_container_width=True, config={"displayModeBar": False})

    # Engagement score distribution
    st.markdown('<p style="font-size:11px;color:#6B7FA8;text-transform:uppercase;'
                'letter-spacing:.06em;font-weight:600;margin-top:4px;">'
                'Engagement score distribution by tier</p>', unsafe_allow_html=True)
    fig5 = go.Figure()
    for tier, color in TIER_COLORS.items():
        sub = df[df["tier"]==tier]
        if len(sub) == 0:
            continue
        fig5.add_trace(go.Histogram(
            x=sub["engagement_score"], name=tier, nbinsx=25,
            marker_color=color, opacity=0.75,
        ))
    fig5.update_layout(**PLOT, height=180, barmode="overlay",
        xaxis=dict(showgrid=False, title="Engagement Score", tickfont=dict(color="#94A3B8")),
        yaxis=dict(showgrid=False, showticklabels=False),
        legend=dict(font=dict(size=11, color="#94A3B8"), bgcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig5, use_container_width=True, config={"displayModeBar": False})

    # SQL query result
    st.markdown('<p style="font-size:11px;color:#6B7FA8;text-transform:uppercase;'
                'letter-spacing:.06em;font-weight:600;margin-top:4px;">'
                'Tier summary — SQL query result</p>', unsafe_allow_html=True)
    tier_sql = pd.read_sql("""
        SELECT
            tier,
            COUNT(*) AS users,
            ROUND(AVG(engagement_score), 1) AS avg_engagement,
            ROUND(AVG(churn_prob)*100, 1) AS avg_churn_risk_pct,
            ROUND(AVG(days_since_active), 0) AS avg_days_inactive,
            SUM(est_roi) AS total_est_roi
        FROM users
        GROUP BY tier
        ORDER BY avg_engagement DESC
    """, conn)
    tier_sql.columns = ["Tier","Users","Avg Engagement","Avg Churn Risk %",
                        "Avg Days Inactive","Total Est. ROI ($)"]
    st.dataframe(tier_sql, use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════
# TAB 3 — OUTREACH ENGINE
# ════════════════════════════════════════════════════════════
with t3:
    st.markdown("##### Ranked Outreach Prioritization — Account Value × Churn Risk × Recency")

    o1, o2, o3 = st.columns(3)
    o1.metric("Users requiring action",
              f"{((df['tier'].isin(['At-Risk','Dormant'])) | (df['tier']=='Power')).sum():,}")
    o2.metric("Total estimated ROI",
              f"${df['est_roi'].sum():,}")
    o3.metric("High-urgency accounts",
              f"{(df.groupby('account')['churn_prob'].mean() > 0.5).sum()}",
              help="Accounts where avg churn risk > 50%")

    st.markdown("<br>", unsafe_allow_html=True)

    # Filters
    fc1, fc2 = st.columns(2)
    with fc1:
        outreach_tier = st.multiselect("Filter by tier",
            ["Power","Active","At-Risk","Dormant"],
            default=["At-Risk","Dormant"],
            key="outreach_tier")
    with fc2:
        max_days = st.slider("Max days inactive", 0, 180, 90, key="max_days")

    # Build ranked list
    outreach_df = df[
        df["tier"].isin(outreach_tier) &
        (df["days_since_active"] <= max_days)
    ].copy()

    # Priority score: weighted rank
    outreach_df["priority_score"] = (
        0.35 * (outreach_df["churn_prob"]) +
        0.35 * (outreach_df["account_value"] / outreach_df["account_value"].max()) +
        0.30 * (outreach_df["days_since_active"] / 180)
    )
    outreach_df = outreach_df.sort_values("priority_score", ascending=False)

    display_out = outreach_df[[
        "account","tier","engagement_score","churn_prob",
        "days_since_active","account_value","recommended_action","est_roi"
    ]].head(50).copy()
    display_out["churn_prob"] = (display_out["churn_prob"]*100).round(1)
    display_out["account_value"] = display_out["account_value"].apply(lambda x: f"${x:,}")
    display_out["est_roi"] = display_out["est_roi"].apply(lambda x: f"${x:,}")
    display_out.columns = ["Account","Tier","Engagement","Churn Risk %",
                           "Days Inactive","Acct Value","Recommended Action","Est. ROI"]

    st.markdown('<p style="font-size:11px;color:#6B7FA8;text-transform:uppercase;'
                'letter-spacing:.06em;font-weight:600;">Top 50 prioritised users</p>',
                unsafe_allow_html=True)
    st.dataframe(display_out, use_container_width=True, hide_index=True)

    # ROI by action type
    st.markdown('<p style="font-size:11px;color:#6B7FA8;text-transform:uppercase;'
                'letter-spacing:.06em;font-weight:600;margin-top:12px;">'
                'Estimated ROI by intervention type</p>', unsafe_allow_html=True)
    roi_by_tier = df.groupby("tier")["est_roi"].agg(["sum","mean","count"]).reset_index()
    roi_by_tier.columns = ["tier","total_roi","avg_roi","users"]
    roi_by_tier = roi_by_tier.sort_values("total_roi", ascending=True)
    fig6 = go.Figure(go.Bar(
        x=roi_by_tier["total_roi"], y=roi_by_tier["tier"], orientation="h",
        marker_color=[TIER_COLORS.get(t, GRAY) for t in roi_by_tier["tier"]],
        text=[f"${v:,.0f}" for v in roi_by_tier["total_roi"]],
        textposition="outside", textfont=dict(size=10, color="#94A3B8"),
    ))
    fig6.update_layout(**PLOT, height=200,
        xaxis=dict(showgrid=False, showticklabels=False),
        yaxis=dict(showgrid=False, tickfont=dict(color="#94A3B8")))
    st.plotly_chart(fig6, use_container_width=True, config={"displayModeBar": False})

    # Export
    csv = outreach_df[["account","tier","engagement_score","churn_prob",
                        "days_since_active","recommended_action","est_roi"]].to_csv(index=False)
    st.download_button("Download outreach list CSV", csv,
                       "userpulse_outreach.csv", "text/csv")


# ════════════════════════════════════════════════════════════
# TAB 4 — COHORT ANALYTICS
# ════════════════════════════════════════════════════════════
with t4:
    st.markdown("##### Cohort Retention Analytics — Week-over-Week Activation Tracking")

    # Cohort retention: users grouped by week since last active
    cohort = df.groupby(["week_cohort","tier"]).size().reset_index(name="users")
    cohort_total = df.groupby("week_cohort").size().reset_index(name="total")
    cohort_active = df[df["tier"].isin(["Power","Active"])].groupby(
        "week_cohort").size().reset_index(name="active")
    cohort_merged = cohort_total.merge(cohort_active, on="week_cohort", how="left").fillna(0)
    cohort_merged["retention_pct"] = (cohort_merged["active"]/cohort_merged["total"]*100).round(1)
    cohort_merged = cohort_merged[cohort_merged["week_cohort"] <= 12]

    ca1, ca2, ca3 = st.columns(3)
    ca1.metric("Week 0 retention",
               f"{cohort_merged[cohort_merged['week_cohort']==0]['retention_pct'].values[0] if len(cohort_merged[cohort_merged['week_cohort']==0]) else 0:.1f}%")
    ca2.metric("Week 4 retention",
               f"{cohort_merged[cohort_merged['week_cohort']==4]['retention_pct'].values[0] if len(cohort_merged[cohort_merged['week_cohort']==4]) else 0:.1f}%")
    ca3.metric("Week 8 retention",
               f"{cohort_merged[cohort_merged['week_cohort']==8]['retention_pct'].values[0] if len(cohort_merged[cohort_merged['week_cohort']==8]) else 0:.1f}%")

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<p style="font-size:11px;color:#6B7FA8;text-transform:uppercase;'
                'letter-spacing:.06em;font-weight:600;">Retention curve — % active users by cohort week</p>',
                unsafe_allow_html=True)
    fig7 = go.Figure()
    fig7.add_trace(go.Scatter(
        x=cohort_merged["week_cohort"],
        y=cohort_merged["retention_pct"],
        mode="lines+markers",
        line=dict(color=INDIGO, width=2.5),
        marker=dict(size=7, color=INDIGO),
        fill="tozeroy", fillcolor="rgba(99,102,241,0.08)",
        name="Active %",
    ))
    fig7.update_layout(**PLOT, height=200,
        xaxis=dict(showgrid=False, title="Weeks since last active",
                   tickfont=dict(color="#94A3B8"),
                   tickvals=list(range(0,13))),
        yaxis=dict(showgrid=False, tickfont=dict(color="#94A3B8"), ticksuffix="%"),
    )
    st.plotly_chart(fig7, use_container_width=True, config={"displayModeBar": False})

    # Stacked tier over time
    st.markdown('<p style="font-size:11px;color:#6B7FA8;text-transform:uppercase;'
                'letter-spacing:.06em;font-weight:600;margin-top:4px;">'
                'Tier composition by cohort week</p>', unsafe_allow_html=True)
    pivot = cohort[cohort["week_cohort"]<=12].pivot(
        index="week_cohort", columns="tier", values="users").fillna(0)
    fig8 = go.Figure()
    for tier in ["Power","Active","At-Risk","Dormant"]:
        if tier in pivot.columns:
            fig8.add_trace(go.Bar(
                x=pivot.index, y=pivot[tier], name=tier,
                marker_color=TIER_COLORS[tier],
            ))
    fig8.update_layout(**PLOT, height=190, barmode="stack",
        xaxis=dict(showgrid=False, title="Weeks since last active",
                   tickfont=dict(color="#94A3B8")),
        yaxis=dict(showgrid=False, tickfont=dict(color="#94A3B8")),
        legend=dict(font=dict(size=11, color="#94A3B8"), bgcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig8, use_container_width=True, config={"displayModeBar": False})

    # Accounts approaching renewal with high churn risk
    st.markdown('<p style="font-size:11px;color:#6B7FA8;text-transform:uppercase;'
                'letter-spacing:.06em;font-weight:600;margin-top:4px;">'
                'High-risk accounts approaching renewal</p>', unsafe_allow_html=True)
    renewal_risk = pd.read_sql("""
        SELECT
            account,
            MIN(days_to_renewal) AS days_to_renewal,
            COUNT(*) AS total_users,
            SUM(CASE WHEN tier IN ('At-Risk','Dormant') THEN 1 ELSE 0 END) AS at_risk_users,
            ROUND(AVG(churn_prob)*100, 1) AS avg_churn_risk_pct,
            ROUND(AVG(engagement_score), 1) AS avg_engagement,
            MAX(account_value) AS account_value
        FROM users
        WHERE days_to_renewal <= 90
        GROUP BY account
        HAVING avg_churn_risk_pct > 40
        ORDER BY avg_churn_risk_pct DESC
        LIMIT 15
    """, conn)
    if len(renewal_risk) > 0:
        renewal_risk["account_value"] = renewal_risk["account_value"].apply(lambda x: f"${x:,}")
        renewal_risk.columns = ["Account","Days to Renewal","Users","At-Risk Users",
                                 "Avg Churn Risk %","Avg Engagement","Account Value"]
        st.dataframe(renewal_risk, use_container_width=True, hide_index=True)
        st.error(
            f"**{len(renewal_risk)} accounts** renewing within 90 days have average churn risk above 40%. "
            f"Prioritise these for immediate success team outreach before renewal conversations begin.",
            icon="🚨"
        )
    else:
        st.info("No high-risk renewal accounts in current filter.", icon="ℹ️")
