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
    all_prompts: typing.Optional[typing.List[str]] = None,
) -> typing.List[typing.Optional["plotly.graph_objects.Figure"]]:
    """
    Visualizes the evaluation results as a timeline using Plotly Express.

    Each prompt is represented as a row, and predicted spans are shown as colored bars.
    Returns one figure per motion in the evaluation result.

    Args:
        result (EvaluationResult): The output from the model's forward + decode methods.
        title (str): The title for the plot.
        all_prompts (List[str], optional): List of all prompts that should be shown in the visualization,
            even if they have no predictions. This ensures empty prompts are still visible on the timeline.

    Returns:
        A list of Plotly Figure objects (one per motion), or None for motions with no valid data.
    """
    figures = []
    
    for motion_idx in range(len(result.motion_length)):
        motion_length = result.motion_length[motion_idx]
        motion_predictions = result.predictions[motion_idx] if motion_idx < len(result.predictions) else []
        
        data_for_dataframe = []
        prompts_with_predictions = set()
        
        for prompt_text, span_list in motion_predictions:
            if span_list:
                prompts_with_predictions.add(prompt_text)
                
                # NOTE: multiple spans for the same prompt
                for span_start, span_end, score in span_list:
                    start_timestamp = dt.datetime(2025, 1, 1, 0, 0, 0) + dt.timedelta(seconds=span_start/DEFAULT_FPS)
                    end_timestamp = dt.datetime(2025, 1, 1, 0, 0, 0) + dt.timedelta(seconds=span_end/DEFAULT_FPS)

                    data_for_dataframe.append({
                        "prompt": prompt_text,
                        "start": span_start,
                        "finish": span_end,
                        "start_timestamp": start_timestamp,
                        "end_timestamp": end_timestamp,
                        "score": f"{score:.2f}"
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
                        "score": "N/A"
                    })

        if len(data_for_dataframe) == 0:
            dataframe = pd.DataFrame({
                "prompt": [],
                "start": [],
                "finish": [],
                "start_timestamp": [],
                "end_timestamp": [],
                "score": []
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

        motion_title = f"{title} - Motion {motion_idx + 1}"
        
        figure = px.timeline(
            dataframe,
            x_start="start_timestamp",
            x_end="end_timestamp",
            y="prompt",
            color="prompt",
            hover_data=hover_data_fields,
            title=motion_title,
            category_orders=category_orders
        )

        # NOTE: set x-axis range to always show full motion duration from frame 0 to motion_length
        motion_start_time = dt.datetime(2025, 1, 1, 0, 0, 0)
        motion_end_time = dt.datetime(2025, 1, 1, 0, 0, 0) + dt.timedelta(seconds=motion_length/DEFAULT_FPS)

        figure.update_layout(
            xaxis_title="Time",
            yaxis_title="Prompt",
            showlegend=False,
            xaxis=dict(range=[motion_start_time, motion_end_time]),
        )

        figure.update_yaxes(autorange="reversed")
        
        figures.append(figure)

    return figures