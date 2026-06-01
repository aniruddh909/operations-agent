"""Typed configuration — the one place tunable knobs and secrets are declared.

Two concerns live here, deliberately separated:

- ``Settings`` — secrets and environment-specific values, loaded from the
  environment / ``.env`` (API keys, Jira host + auth, the project key). These
  are required for *live* runs and fail fast with a clear, named error when
  missing.
- Plain config values with sensible defaults (model IDs, and — in later slices —
  the confidence threshold and cosine bands) also live on ``Settings`` so there
  is a single typed surface a reviewer can read to see everything that's tunable.

Nothing here imports the agent loop, so importing config is cheap and safe from
the CLI before any heavy work begins.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class MissingConfigError(RuntimeError):
    """Raised when a required setting for a live run is absent.

    The message names the missing environment variable(s) so first-run setup is
    a copy-paste away.
    """


class Settings(BaseSettings):
    """All configuration for the agent, typed and documented in one place."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # -- Secrets / environment (required for live runs) -- #
    anthropic_api_key: str | None = Field(
        default=None, description="Anthropic API key. Env: ANTHROPIC_API_KEY"
    )
    jira_base_url: str | None = Field(
        default=None,
        description="Jira site base URL, e.g. https://you.atlassian.net. "
        "Env: JIRA_BASE_URL",
    )
    jira_email: str | None = Field(
        default=None,
        description="Atlassian account email for API auth. Env: JIRA_EMAIL",
    )
    jira_api_token: str | None = Field(
        default=None,
        description="Atlassian API token. Env: JIRA_API_TOKEN",
    )

    # -- Tunable knobs (safe defaults) -- #
    jira_project_key: str = Field(
        default="OPS",
        description="Project key new tickets are filed under. Env: JIRA_PROJECT_KEY",
    )
    planning_model: str = Field(
        default="claude-sonnet-4-6",
        description="Model for planning / reasoning / adjudication.",
    )
    judge_model: str = Field(
        default="claude-haiku-4-5-20251001",
        description="Cheap model for LLM-as-judge and routine classification.",
    )

    # -- Duplicate detection knobs -- #
    index_path: str = Field(
        default="ticket_index.db",
        description="SQLite file backing the local ticket vector index.",
    )
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Local sentence-transformers model for embeddings.",
    )
    dup_clear_band: float = Field(
        default=0.75,
        description="Cosine at/above this + model agrees = a clear duplicate.",
    )
    dup_ambiguous_band: float = Field(
        default=0.55,
        description="Cosine in [ambiguous, clear) is an ambiguous duplicate "
        "(Slice 4 asks a human). Below this = novel bug.",
    )

    def require_live(self) -> None:
        """Fail fast (naming the gaps) if anything needed for a live run is unset."""
        required = {
            "ANTHROPIC_API_KEY": self.anthropic_api_key,
            "JIRA_BASE_URL": self.jira_base_url,
            "JIRA_EMAIL": self.jira_email,
            "JIRA_API_TOKEN": self.jira_api_token,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise MissingConfigError(
                "Missing required configuration for a live run: "
                + ", ".join(missing)
                + ". Copy .env.example to .env and fill these in."
            )
