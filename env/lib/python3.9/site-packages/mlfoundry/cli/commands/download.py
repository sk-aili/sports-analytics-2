import click

from mlfoundry import get_client


@click.group(help="Download model logged with Mlfoundry")
def download():
    ...


@download.command(short_help="Download a logged model")
@click.option(
    "--fqn",
    required=True,
    type=str,
    help="fqn of the model version",
)
@click.option(
    "--path",
    type=click.Path(file_okay=False, dir_okay=True, exists=False),
    required=True,
    help="path where the model will be downloaded",
)
def model(fqn: str, path: str):
    """
    Download the logged model for a run.\n
    """
    client = get_client()
    model_version = client.get_model(fqn=fqn)
    download_path = model_version.download(path=path)
    print(f"Downloaded model files to {download_path}")
