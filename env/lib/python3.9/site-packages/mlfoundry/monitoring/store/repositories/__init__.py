from mlfoundry.monitoring.store.repositories.rest_monitoring_store import (
    RestMonitoringStore,
)
from mlfoundry.session import Session


def get_monitoring_store(session: Session):
    # For now we have only one repo
    return RestMonitoringStore(
        get_host_creds=session.get_monitoring_foundry_host_creds_builder()
    )
