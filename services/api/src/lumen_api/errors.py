"""RFC 9457 problem+json errors.

One shape for every failure the API produces, so the web client has exactly one
error branch to write and one field (`detail`) to show a person.
"""

from __future__ import annotations

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel


class ProblemDetail(BaseModel):
    type: str = "about:blank"
    title: str
    status: int
    detail: str | None = None


class AppError(Exception):
    """Base for errors the API converts into problem+json rather than a 500."""

    status_code: int = status.HTTP_400_BAD_REQUEST
    title: str = "Request failed"

    def __init__(self, detail: str | None = None) -> None:
        super().__init__(detail or self.title)
        self.detail = detail


class BadRequest(AppError):
    status_code = status.HTTP_400_BAD_REQUEST
    title = "Bad request"


class Unauthorized(AppError):
    status_code = status.HTTP_401_UNAUTHORIZED
    title = "Not authenticated"


class Forbidden(AppError):
    status_code = status.HTTP_403_FORBIDDEN
    title = "Not permitted"


class NotFound(AppError):
    status_code = status.HTTP_404_NOT_FOUND
    title = "Not found"


class Conflict(AppError):
    status_code = status.HTTP_409_CONFLICT
    title = "Conflict"


class UnsupportedMedia(AppError):
    status_code = status.HTTP_415_UNSUPPORTED_MEDIA_TYPE
    title = "Unsupported file type"


class QuotaExceeded(AppError):
    status_code = status.HTTP_429_TOO_MANY_REQUESTS
    title = "Quota exceeded"


class TooManyRequests(AppError):
    """Distinct from `QuotaExceeded`: a rate limit is about request pace
    (ADR-0009's trust API, per key), not a plan's monthly usage allowance."""

    status_code = status.HTTP_429_TOO_MANY_REQUESTS
    title = "Rate limit exceeded"


class Misconfigured(AppError):
    """A credential or setting the operator must fix — never the caller's fault."""

    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    title = "Service not configured"


def _problem(status_code: int, title: str, detail: str | None) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=ProblemDetail(title=title, status=status_code, detail=detail).model_dump(),
        media_type="application/problem+json",
    )


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def _app_error(_: Request, exc: AppError) -> JSONResponse:
        return _problem(exc.status_code, exc.title, exc.detail)

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        first = exc.errors()[0] if exc.errors() else {}
        location = ".".join(str(part) for part in first.get("loc", ())[1:]) or "body"
        return _problem(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "Validation failed",
            f"{location}: {first.get('msg', 'invalid value')}",
        )
