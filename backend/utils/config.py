from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    groq_api_key: str = ""
    sarvam_api_key: str = ""
    langchain_api_key: str = ""
    langchain_tracing_v2: bool = True
    langchain_project: str = "clinic-voice-agent"
    database_url: str = ""
    redis_url: str = ""
    google_calendar_credentials_json: str = ""


config = Settings()
