import enum
import json
import os
import posixpath
import sys
import tempfile
from typing import Union

from mlflow.entities import ArtifactType
from pydantic import BaseModel

from mlfoundry.exceptions import MlFoundryException
from mlfoundry.log_types.artifacts.artifact import (
    ArtifactVersionInternalMetadata,
    _log_artifact_version_helper,
)
from mlfoundry.log_types.artifacts.constants import FILES_DIR, INTERNAL_METADATA_PATH
from mlfoundry.log_types.pydantic_base import PydanticBase
from mlfoundry.log_types.utils import validate_key_name

PlotObjType = Union[
    "matplotlib.figure.Figure",
    "plotly.graph_objects.Figure",
    "matplotlib.pyplot",
]


@enum.unique
class Format(enum.Enum):
    SVG = "SVG"
    HTML = "HTML"
    PNG = "PNG"


def _is_matplotlib_figure(fig) -> bool:
    if "matplotlib" not in sys.modules:
        return False
    import matplotlib

    return isinstance(fig, matplotlib.figure.Figure)


def _is_matplotlib_plt(plt) -> bool:
    if "matplotlib" not in sys.modules:
        return False
    return getattr(plt, "__name__", "") == "matplotlib.pyplot"


def _is_plotly_figure(fig) -> bool:
    if "plotly" not in sys.modules:
        return False

    import plotly

    return isinstance(fig, plotly.graph_objects.Figure)


def get_plot_file_name(format: Format) -> str:
    return f"plot.{format.value.lower()}"


class PlotArtifact(BaseModel):
    artifact_file: str
    format: Format

    class Config:
        allow_mutation = False
        use_enum_values = True


def _save_matplotlib_figure(
    figure: "matplotlib.figure.Figure",
    key: str,
    step: int,
    local_dir: str,
) -> PlotArtifact:
    supported_formats = figure.canvas.get_supported_filetypes().keys()
    if "svg" in supported_formats:
        format_ = Format.SVG
    elif "png" in supported_formats:
        format_ = Format.PNG
    else:
        raise MlFoundryException(
            f"Could not save {key} {figure} matplotlib figure"
            "in either SVG or PNG format"
        )
    file_path = get_plot_file_name(format=format_)
    local_path = os.path.join(local_dir, file_path)
    figure.savefig(local_path)
    return PlotArtifact(artifact_file=file_path, format=format_)


def _save_matplotlib_plt(
    plt: "matplotlib.pyplot",
    key: str,
    step: int,
    local_dir: str,
) -> PlotArtifact:
    figure = plt.gcf()
    return _save_matplotlib_figure(
        figure=figure, key=key, step=step, local_dir=local_dir
    )


def _save_plotly_figure(
    figure: "plotly.graph_objects.Figure",
    key: str,
    step: int,
    local_dir: str,
) -> PlotArtifact:
    format_ = Format.HTML
    file_path = get_plot_file_name(format=format_)
    local_path = os.path.join(local_dir, file_path)
    figure.write_html(local_path, include_plotlyjs="cdn", auto_open=False)
    return PlotArtifact(artifact_file=file_path, format=format_)


class Plot:
    def __init__(self, plot_obj: PlotObjType):
        self._plot_obj = plot_obj

    def _save_plot(self, key: str, step: int, local_dir: str) -> PlotArtifact:
        if _is_matplotlib_plt(self._plot_obj):
            return _save_matplotlib_plt(
                plt=self._plot_obj, key=key, step=step, local_dir=local_dir
            )

        if _is_matplotlib_figure(self._plot_obj):
            return _save_matplotlib_figure(
                figure=self._plot_obj, key=key, step=step, local_dir=local_dir
            )

        if _is_plotly_figure(self._plot_obj):
            return _save_plotly_figure(
                figure=self._plot_obj, key=key, step=step, local_dir=local_dir
            )

        raise MlFoundryException(
            f"Unknown type: {type(self._plot_obj)}"
            "Supported types are, matplotlib.figure.Figure, matplotlib.pyplot"
            " and plotly.graph_objects.Figure"
        )

    def save(
        self,
        run: "mlfoundry.MlFoundryRun",
        key: str,
        step: int = 0,
    ):
        validate_key_name(key)

        # creating a temp dir which will be logged
        temp_dir = tempfile.TemporaryDirectory(prefix="truefoundry-")
        local_files_dir = os.path.join(temp_dir.name, FILES_DIR)
        os.makedirs(local_files_dir, exist_ok=True)

        # save plot locally
        plot_artifact = self._save_plot(key=key, step=step, local_dir=local_files_dir)

        # save internal metadata
        internal_metadata = PlotVersionInternalMetadata(
            files_dir=FILES_DIR,
            plot_file=posixpath.join(FILES_DIR, plot_artifact.artifact_file),
        )
        local_internal_metadata_path = os.path.join(
            temp_dir.name, INTERNAL_METADATA_PATH
        )
        os.makedirs(os.path.dirname(local_internal_metadata_path), exist_ok=True)
        with open(local_internal_metadata_path, "w") as f:
            json.dump(internal_metadata.dict(), f)

        # log the artifact
        _log_artifact_version_helper(
            run=run,
            name=key,
            artifact_type=ArtifactType.PLOT,
            artifact_dir=temp_dir,
            step=step,
        )


class PlotRunLogType(PydanticBase):
    value: PlotArtifact

    @staticmethod
    def get_log_type() -> str:
        return "plot"


class PlotVersionInternalMetadata(ArtifactVersionInternalMetadata):
    plot_file: str


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    import plotly.express as px

    import mlfoundry

    client = mlfoundry.get_client("https://app.devtest.truefoundry.tech")
    # client = mlfoundry.get_client("http://localhost:5000")

    run = client.create_run(project_name="plot-test1")

    df = px.data.iris()
    fig = px.scatter(
        df,
        x="sepal_width",
        y="sepal_length",
        color="species",
        size="petal_length",
        hover_data=["petal_width"],
    )
    Plot(fig).save(run, "foo")

    df = px.data.tips()
    fig = px.histogram(
        df,
        x="total_bill",
        y="tip",
        color="sex",
        marginal="rug",
        hover_data=df.columns,
    )
    Plot(fig).save(run, "foo", step=1)

    names = ["group_a", "group_b", "group_c"]
    values = [1, 10, 100]
    plt.figure(figsize=(9, 3))
    plt.subplot(131)
    plt.bar(names, values)
    plt.subplot(132)
    plt.scatter(names, values)
    plt.subplot(133)
    plt.plot(names, values)
    plt.suptitle("Categorical Plotting")

    Plot(plt).save(run, "bar")

    plt.clf()

    data = {
        "a": np.arange(50),
        "c": np.random.randint(0, 50, 50),
        "d": np.random.randn(50),
    }
    data["b"] = data["a"] + 10 * np.random.randn(50)
    data["d"] = np.abs(data["d"]) * 100
    plt.scatter("a", "b", c="c", s="d", data=data)
    plt.xlabel("entry a")
    plt.ylabel("entry b")
    Plot(plt).save(run, "bar", 2)

    plt.clf()

    ax = plt.subplot()
    t = np.arange(0.0, 5.0, 0.01)
    s = np.cos(2 * np.pi * t)
    (line,) = plt.plot(t, s, lw=2)

    plt.annotate(
        "local max",
        xy=(2, 1),
        xytext=(3, 1.5),
        arrowprops=dict(facecolor="black", shrink=0.05),
    )

    plt.ylim(-2, 2)
    Plot(plt.gcf()).save(run, "bar", 3)
