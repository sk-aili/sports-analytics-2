import click


@click.group("db")
def commands():
    """
    Commands for managing an MLflow tracking database.
    """
    pass


@commands.command()
@click.argument("url")
def upgrade(url):
    """
    Upgrade the schema of an MLflow tracking database to the latest supported version.

    **IMPORTANT**: Schema migrations can be slow and are not guaranteed to be transactional -
    **always take a backup of your database before running migrations**. The migrations README,
    which is located at
    https://github.com/mlflow/mlflow/blob/master/mlflow/store/db_migrations/README.md, describes
    large migrations and includes information about how to estimate their performance and
    recover from failures.
    """
    import mlflow.store.db.utils

    engine = mlflow.store.db.utils.create_sqlalchemy_engine_with_retry(url)

    # The alembic migrations do not work on a fresh database
    # Some tables are brought up using in the tracking sqlalchemy
    # store using _initialize_tables function. _initialize_tables,
    # creates those tables and calls _upgrade_db (alembic migration) after that.
    # Here, I am calling _initialize_tables directly for now.

    # mlflow.store.db.utils._upgrade_db(engine)
    mlflow.store.db.utils._initialize_tables(engine)
