from pathlib import Path

import click
import pandas as pd

from pandas_profiling import ProfileReport

@click.command()
@click.option(
    "-d",
    "--dataset-path",
    default="data/train.csv",
    type=click.Path(exists=True,
                    dir_okay=False,
                    path_type=Path),
    show_default=True
)
@click.option(
    "-s",
    "--save-report-path",
    default="report/eda.html",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    show_default=True,
)

def create_eda(
    dataset_path: Path,
    save_report_path: Path
) -> None:
    path_folder = save_report_path.parent
    path_folder.mkdir(exist_ok=True)
    save_report_path.unlink(missing_ok=True)
    data = pd.read_csv(dataset_path, index_col='Id')
    profile = ProfileReport(data, title="Pandas Profiling Report")
    profile.to_file(save_report_path)

    click.echo(f"Report is saved to {save_report_path}.")