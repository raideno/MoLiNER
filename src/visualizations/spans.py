import typing

import pandas as pd
import datetime as dt
import plotly.express as px

from src.constants import DEFAULT_FPS
from src.data.typing import EvaluationResult

def plot_evaluation_results(
    result: EvaluationResult,
    title: str = "Motion-Prompt Localization Results",
    fps: int = DEFAULT_FPS
) -> typing.Optional["plotly.graph_objects.Figure"]:
    """
    Visualizes the evaluation results as a timeline using Plotly Express.

    Each prompt is represented as a row, and predicted spans are shown as colored bars.

    Args:
        result (EvaluationResult): The output from the model's `evaluate` method.
        title (str): The title for the plot.
        fps (int): Frames per second, used to convert frame numbers to timestamps.

    Returns:
        A Plotly Figure object, or None if there are no predictions.
    """
    if not result.predictions:
        print("No predictions to plot.")
        return None

    data_for_dataframe = []
    
    for prompt, start, end, score in result.predictions:
        start_timestamp = dt.datetime(2025, 1, 1, 0, 0, 0) + dt.timedelta(seconds=start/fps)
        end_timestamp = dt.datetime(2025, 1, 1, 0, 0, 0) + dt.timedelta(seconds=end/fps)
        
        data_for_dataframe.append({
            "prompt": prompt,
            "start": start,
            "finish": end,
            "start_timestamp": start_timestamp,
            "end_timestamp": end_timestamp,
            "score": f"{score:.2f}"
        })
        
    dataframe = pd.DataFrame(data_for_dataframe)
    
    category_orders = {"prompt": sorted(list(dataframe['prompt'].unique()))}

    figure = px.timeline(
        dataframe,
        x_start="start_timestamp",
        x_end="end_timestamp",
        y="prompt",
        color="prompt",
        hover_data=["score", "start", "finish"],
        title=title,
        category_orders=category_orders
    )

    figure.update_layout(
        xaxis_title="Time",
        yaxis_title="Prompt",
        showlegend=False
    )
    
    figure.update_yaxes(autorange="reversed")
    
    return figure