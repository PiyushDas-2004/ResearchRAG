"""FastAPI dependencies: container access, authentication and rate limiting."""

from __future__ import annotations

from typing import Annotated, cast

from fastapi import Depends, Request

from researchrag.api.container import AppContainer
from researchrag.api.security import Principal


def get_container(request: Request) -> AppContainer:
    """The application container built in the lifespan."""
    return cast(AppContainer, request.app.state.container)


def require_principal(
    request: Request, container: Annotated[AppContainer, Depends(get_container)]
) -> Principal:
    """Authenticate the caller and apply the per-key rate limit."""
    principal = container.auth.authenticate(request)
    request.state.key_id = principal.key_id
    container.limiter.check(principal)
    return principal


ContainerDep = Annotated[AppContainer, Depends(get_container)]
PrincipalDep = Annotated[Principal, Depends(require_principal)]
