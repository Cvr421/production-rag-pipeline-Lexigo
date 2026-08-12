from fastapi import FastAPI, HTTPException

from .config import get_settings
from .schemas import AskRequest, AskResponse
from .service import QAService


def create_app() -> FastAPI:
    app = FastAPI(title="Lexigo Document Q&A API", version="0.1.0")
    service = QAService(get_settings())

    @app.get("/health")
    def health() -> dict[str, str | bool]:
        settings = get_settings()
        return {
            "status": "ok",
            "cerebras_configured": bool(settings.cerebras_api_key),
            "pinecone_configured": bool(settings.pinecone_api_key),
        }

    @app.post("/ask", response_model=AskResponse)
    def ask(request: AskRequest) -> AskResponse:
        try:
            result = service.ask(request.question.strip(), request.include_trace)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=502, detail="Q&A backend request failed") from exc
        return AskResponse(**result)

    return app


app = create_app()

