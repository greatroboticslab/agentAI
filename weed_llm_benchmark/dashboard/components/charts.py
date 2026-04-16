"""Reusable Plotly chart builders."""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


COLORS = {
    "old": "#2196F3",      # blue
    "new": "#FF9800",      # orange
    "baseline": "#4CAF50", # green
    "forgetting": "#F44336", # red
    "ok": "#4CAF50",
    "threshold": "#F44336",
}

TOOL_COLORS = {
    "inspect_labels": "#9E9E9E",
    "run_vlm_inference": "#03A9F4",
    "generate_consensus": "#8BC34A",
    "filter_labels": "#FF9800",
    "train_yolo": "#2196F3",
    "freeze_train": "#3F51B5",
    "lora_train": "#9C27B0",
    "distill_train": "#673AB7",
    "two_pass_train": "#E91E63",
    "evaluate": "#4CAF50",
    "analyze_failure": "#F44336",
    "search_models": "#00BCD4",
    "run_external_model": "#009688",
    "identify_weed": "#FF5722",
    "done": "#607D8B",
    "ROUND_START": "#000000",
    "EVAL_RESULT": "#4CAF50",
    "FILTER_RESULT": "#FF9800",
    "LORA_INJECT": "#9C27B0",
}


def metric_line_chart(experiments, y_cols, title, threshold=None):
    """Line chart of metrics over iterations."""
    if not experiments:
        return go.Figure().update_layout(title=title)

    df = pd.DataFrame(experiments)
    fig = go.Figure()

    for col in y_cols:
        if col in df.columns:
            color = COLORS.get("old" if "old" in col else "new", "#333")
            name = col.replace("_", " ").title()
            fig.add_trace(go.Scatter(
                x=df["iteration"], y=df[col], mode="lines+markers",
                name=name, line=dict(color=color, width=2),
                marker=dict(size=8),
                hovertemplate=f"{name}: %{{y:.4f}}<br>Iter: %{{x}}<extra></extra>",
            ))

    if threshold is not None:
        fig.add_hline(y=threshold, line_dash="dash", line_color=COLORS["threshold"],
                      annotation_text=f"Threshold: {threshold}")

    fig.update_layout(
        title=title, xaxis_title="Iteration", yaxis_title="Score",
        template="plotly_white", height=400, legend=dict(orientation="h", y=-0.15),
    )
    return fig


def comparison_bar_chart(data, title):
    """Grouped bar chart for model comparison."""
    if not data:
        return go.Figure().update_layout(title=title)

    df = pd.DataFrame(data)
    fig = go.Figure()

    metrics = [c for c in df.columns if c not in ("method", "params")]
    for metric in metrics:
        fig.add_trace(go.Bar(name=metric.replace("_", " ").title(),
                             x=df["method"], y=df[metric]))

    fig.update_layout(
        title=title, barmode="group", template="plotly_white",
        height=400, legend=dict(orientation="h", y=-0.15),
    )
    return fig


def action_timeline(actions, title="Brain Action Timeline"):
    """Horizontal timeline of Brain actions."""
    if not actions:
        return go.Figure().update_layout(title=title)

    categories = []
    colors = []
    texts = []
    for i, a in enumerate(actions):
        action = a.get("action", "?")
        categories.append(action)
        colors.append(TOOL_COLORS.get(action, "#999"))
        reasoning = a.get("reasoning", a.get("detail", ""))
        texts.append(f"{action}<br>{reasoning[:60]}")

    fig = go.Figure(go.Bar(
        y=list(range(len(actions))), x=[1] * len(actions),
        orientation="h", marker_color=colors,
        text=texts, textposition="inside",
        hovertemplate="%{text}<extra></extra>",
    ))

    fig.update_layout(
        title=title, template="plotly_white", height=max(300, len(actions) * 35),
        yaxis=dict(tickvals=list(range(len(actions))),
                   ticktext=[f"Step {i+1}" for i in range(len(actions))],
                   autorange="reversed"),
        xaxis=dict(visible=False),
        showlegend=False,
    )
    return fig


def tool_usage_pie(actions):
    """Pie chart of tool usage frequency."""
    if not actions:
        return go.Figure()

    counts = {}
    for a in actions:
        action = a.get("action", "?")
        if action in ("ROUND_START", "EVAL_RESULT", "FILTER_RESULT", "LORA_INJECT"):
            continue
        counts[action] = counts.get(action, 0) + 1

    fig = go.Figure(go.Pie(
        labels=list(counts.keys()), values=list(counts.values()),
        marker_colors=[TOOL_COLORS.get(k, "#999") for k in counts.keys()],
        textinfo="label+percent",
    ))
    fig.update_layout(title="Tool Usage Distribution", height=350)
    return fig


def forgetting_scatter(experiments):
    """Scatter plot: old_f1 vs new_f1, colored by forgetting status."""
    if not experiments:
        return go.Figure()

    df = pd.DataFrame(experiments)
    fig = px.scatter(
        df, x="old_f1", y="new_f1", color="forgetting",
        color_discrete_map={True: COLORS["forgetting"], False: COLORS["ok"]},
        hover_data=["name", "old_map50", "new_map50"],
        title="Old F1 vs New F1 (Forgetting Analysis)",
        labels={"old_f1": "Old Species F1", "new_f1": "New Species F1"},
    )
    fig.add_vline(x=0.90, line_dash="dash", line_color=COLORS["threshold"],
                  annotation_text="Forgetting Threshold")
    fig.update_layout(template="plotly_white", height=400)
    return fig
