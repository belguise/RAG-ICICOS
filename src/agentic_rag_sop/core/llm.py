"""
Gemini LLM Provider.
Initializes ChatGoogleGenerativeAI with strict parameter defaults
for zero-hallucination procedural question answering.
"""

from typing import Optional
from langchain_core.language_models import BaseChatModel

from agentic_rag_sop.config.settings import get_settings


def get_llm(
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
    streaming: bool = False,
) -> BaseChatModel:
    """
    Initialize and return Google Gemini Chat Model.
    
    Args:
        model_name: Model identifier (defaults to settings.GEMINI_MODEL, e.g. gemini-1.5-flash)
        temperature: Sampling temperature (defaults to settings.LLM_TEMPERATURE, 0.0 for deterministic answers)
        streaming: Whether to stream tokens
    """
    settings = get_settings()
    api_key = settings.google_api_key

    if not api_key:
        raise ValueError(
            "Google Gemini API key not found.\n"
            "Please open your '.env' file in the project root and add your key:\n"
            "  GEMINI_API_KEY=your_gemini_api_key_here\n"
            "You can obtain an API key from Google AI Studio: https://aistudio.google.com/app/apikey"
        )

    raw_model = model_name or settings.GEMINI_MODEL
    # Strip optional 'models/' prefix for API compatibility
    target_model = raw_model.replace("models/", "").strip()

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI

        primary_llm = ChatGoogleGenerativeAI(
            model=target_model,
            temperature=temperature if temperature is not None else settings.LLM_TEMPERATURE,
            api_key=api_key,
            google_api_key=api_key,
            max_retries=2,
            streaming=streaming,
        )

        # Resilient fallbacks to handle transient 429 quota exhaustion or 503 outages
        fallback_models = ["gemini-3.5-flash-lite"]
        fallbacks = [
            ChatGoogleGenerativeAI(
                model=m,
                temperature=temperature if temperature is not None else settings.LLM_TEMPERATURE,
                api_key=api_key,
                google_api_key=api_key,
                max_retries=2,
                streaming=streaming,
            )
            for m in fallback_models if m != target_model
        ]

        return primary_llm.with_fallbacks(fallbacks)
    except ImportError as e:
        raise ImportError(
            "Could not import ChatGoogleGenerativeAI. "
            "Ensure langchain-google-genai is installed."
        ) from e
