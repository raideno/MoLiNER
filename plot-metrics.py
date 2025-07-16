import os
import logging
import argparse

import pandas as pd
import matplotlib.pyplot as plt

from hydra import main
from omegaconf import DictConfig

from src.constants import (
    DEFAULT_HYDRA_CONFIG_PATH,
    DEFAULT_HYDRA_VERSION_BASE,
)

logger = logging.getLogger(__name__)

@main(config_path=DEFAULT_HYDRA_CONFIG_PATH, config_name="plot-metrics", version_base=DEFAULT_HYDRA_VERSION_BASE)
def plot_metrics(cfg: DictConfig):
    """
    Reads metrics.csv from the specified run directory, and for each metric,
    plots its evolution over steps and epochs, saving the plots in a 'plots'
    subdirectory within the run directory.
    """
    
    run_dir = cfg.run_dir
    
    metrics_file = os.path.join(run_dir, "logs", "metrics.csv")
    if not os.path.exists(metrics_file):
        logger.error(f"{metrics_file} not found.")
        return

    plots_dir = os.path.join(run_dir, "logs", "plots")
    os.makedirs(plots_dir, exist_ok=True)

    dataframe = pd.read_csv(metrics_file)

    x_cols = [col for col in ['epoch', 'step'] if col in dataframe.columns]
    if not x_cols:
        logger.error("'epoch' or 'step' column not found in metrics.csv.")
        return
        
    metric_cols = [col for col in dataframe.columns if col not in x_cols]

    for metric in metric_cols:
        for x_col in x_cols:
            plot_dataframe = dataframe[[x_col, metric]].dropna()

            if not plot_dataframe.empty:
                plt.figure(figsize=(10, 6))
                plt.plot(plot_dataframe[x_col], plot_dataframe[metric], marker='o')
                
                title = f'{metric} vs {x_col}'
                plt.title(title)
                plt.xlabel(x_col.capitalize())
                plt.ylabel(metric)
                
                filename = f'{metric.replace("/", "_")}_vs_{x_col}.png'
                filepath = os.path.join(plots_dir, filename)
                
                plt.savefig(filepath)
                plt.close()
                logger.info(f"Saved plot to {filepath}")

if __name__ == '__main__':
    plot_metrics()