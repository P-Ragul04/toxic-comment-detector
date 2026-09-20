"""Request/response schemas for the toxic comment detection API."""

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Comment text to classify.",
        examples=["You are an idiot and I hope bad things happen to you."],
    )


class LabelScore(BaseModel):
    label: str
    probability: float
    flagged: bool  # whether this label's tuned threshold was crossed


class PredictResponse(BaseModel):
    text: str
    is_toxic: bool  # true if ANY label was flagged
    scores: list[LabelScore]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class TanglishPredictResponse(BaseModel):
    text: str
    is_offensive: bool
    probability: float  # P(offensive)
