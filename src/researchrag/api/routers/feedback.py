"""Answer feedback."""

from __future__ import annotations

from fastapi import APIRouter, status

from researchrag.api.deps import ContainerDep, PrincipalDep
from researchrag.api.schemas import FeedbackIn, FeedbackOut

router = APIRouter(prefix="/v1", tags=["feedback"])


@router.post(
    "/feedback", status_code=status.HTTP_201_CREATED, response_model=FeedbackOut, summary="Rate an answer"
)
async def post_feedback(body: FeedbackIn, container: ContainerDep, principal: PrincipalDep) -> FeedbackOut:
    """Store thumbs up/down (and an optional comment) against a query trace."""
    rating = 1 if body.rating == "up" else -1
    feedback_id = await container.feedback.add(
        body.trace_id, rating, body.comment, None if principal.anonymous else principal.key_id
    )
    return FeedbackOut(id=feedback_id, trace_id=body.trace_id, rating=body.rating)
