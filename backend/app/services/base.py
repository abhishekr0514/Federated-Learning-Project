"""Base Application Service class for business logic orchestration."""

from typing import Generic, TypeVar
from backend.app.repositories.base import BaseRepository

RepoType = TypeVar("RepoType", bound=BaseRepository)


class BaseService(Generic[RepoType]):
    """Base service encapsulating application workflow logic independent of HTTP protocols."""

    def __init__(self, repository: RepoType):
        self.repository = repository
