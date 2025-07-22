import typing

import pandas as pd
import datetime as dt
import plotly.express as px
import plotly.graph_objects

from src.constants import (
    DEFAULT_FPS
)
from src.types import EvaluationResult

def plot_evaluation_results(
    result: EvaluationResult,
    title: str = "Motion-Prompt Localization Results",
    sources: typing.Optional[typing.List[str]] = None,
    filter_sources: typing.Optional[typing.List[str]] = None,
    all_prompts: typing.Optional[typing.List[str]] = None,
) -> typing.Optional["plotly.graph_objects.Figure"]:
    """
    Visualizes the evaluation results as a timeline using Plotly Express.

    Each prompt is represented as a row, and predicted spans are shown as colored bars.

    Args:
        result (EvaluationResult): The output from the model's forward + decode methods.
        title (str): The title for the plot.
        sources (List[str], optional): List of source labels for each span in result.predictions.
            Must have the same length as result.predictions if provided.
        filter_sources (List[str], optional): If provided, only spans with sources in this list will be displayed.
        all_prompts (List[str], optional): List of all prompts that should be shown in the visualization,
            even if they have no predictions. This ensures empty prompts are still visible on the timeline.

    Returns:
        A Plotly Figure object, or None if there are no predictions.
    """
    data_for_dataframe = []

    if sources is not None and len(sources) != len(result.predictions):
        raise ValueError(f"Sources list length ({len(sources)}) must match predictions length ({len(result.predictions)})")

    prompts_with_predictions = set()
    
    for i, (prompt, start, end, score) in enumerate(result.predictions):
        source = sources[i] if sources is not None else "unknown"

        # NOTE: apply source filtering if specified
        if filter_sources is not None and source not in filter_sources:
            continue

        prompts_with_predictions.add(prompt)
        start_timestamp = dt.datetime(2025, 1, 1, 0, 0, 0) + dt.timedelta(seconds=start/DEFAULT_FPS)
        end_timestamp = dt.datetime(2025, 1, 1, 0, 0, 0) + dt.timedelta(seconds=end/DEFAULT_FPS)

        data_for_dataframe.append({
            "prompt": prompt,
            "start": start,
            "finish": end,
            "start_timestamp": start_timestamp,
            "end_timestamp": end_timestamp,
            "score": f"{score:.2f}",
            "source": source
        })

    # NOTE: ensure all prompts are visible, even if they have no predictions
    if all_prompts is not None:
        motion_start_time = dt.datetime(2025, 1, 1, 0, 0, 0)
        for prompt in all_prompts:
            if prompt not in prompts_with_predictions:
                data_for_dataframe.append({
                    "prompt": prompt,
                    "start": 0,
                    "finish": 0,
                    "start_timestamp": motion_start_time,
                    "end_timestamp": motion_start_time,
                    "score": "N/A",
                    "source": "empty"
                })

    if len(data_for_dataframe) == 0:
        dataframe = pd.DataFrame({
            "prompt": [],
            "start": [],
            "finish": [],
            "start_timestamp": [],
            "end_timestamp": [],
            "score": [],
            "source": []
        })
    else:
        dataframe = pd.DataFrame(data_for_dataframe)

    if all_prompts is not None:
        all_unique_prompts = sorted(all_prompts)
    else:
        if len(dataframe) > 0:
            all_unique_prompts = sorted(dataframe['prompt'].drop_duplicates().tolist())
        else:
            all_unique_prompts = []
    
    category_orders = {"prompt": all_unique_prompts}

    hover_data_fields = ["score", "start", "finish"]
    if sources is not None:
        hover_data_fields.append("source")

    figure = px.timeline(
        dataframe,
        x_start="start_timestamp",
        x_end="end_timestamp",
        y="prompt",
        color="prompt",
        hover_data=hover_data_fields,
        title=title,
        category_orders=category_orders
    )

    # NOTE: set x-axis range to always show full motion duration from frame 0 to motion_length
    motion_start_time = dt.datetime(2025, 1, 1, 0, 0, 0)
    motion_end_time = dt.datetime(2025, 1, 1, 0, 0, 0) + dt.timedelta(seconds=result.motion_length/DEFAULT_FPS)

    figure.update_layout(
        xaxis_title="Time",
        yaxis_title="Prompt",
        showlegend=False,
        xaxis=dict(range=[motion_start_time, motion_end_time]),
    )

    figure.update_yaxes(autorange="reversed")

    return figure