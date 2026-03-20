"""Offline storage stage."""

from backend.offline.storage.local_deploy import deploy_local_artifacts
from backend.offline.storage.redis_ingest import ingest_user_features_to_redis

__all__ = ["deploy_local_artifacts", "ingest_user_features_to_redis"]
