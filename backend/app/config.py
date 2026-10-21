from functools import lru_cache
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True, extra="ignore")

    # API
    API_TITLE: str = "TerraPulse AI"
    API_VERSION: str = "2.0.0"
    DEBUG: bool = False
    CORS_ORIGINS: str = "*"
    APP_TZ: str = "Asia/Kolkata"
    LOG_LEVEL: str = "INFO"
    REQUEST_TIMEOUT_S: float = 90.0

    # Small language model (≤3B, open weights, run locally)
    SLM_ENABLED: bool = True
    SLM_BACKEND: str = "llama_cpp"                 # llama_cpp | transformers | none
    SLM_REPO: str = "aaronr43210/terrapulse-qwen3-1.7b-GGUF"
    SLM_FILE: str = "terrapulse-qwen3-1.7b.Q4_K_M.gguf"
    SLM_BASE_REPO: str = "unsloth/Qwen3-1.7B-GGUF"
    SLM_BASE_FILE: str = "Qwen3-1.7B-Q4_K_M.gguf"
    SLM_HF_MODEL: str = "aaronr43210/terrapulse-qwen3-1.7b"   # transformers backend (from_pretrained)
    SLM_LOCAL_PATH: str = ""                         # optional explicit GGUF path
    SLM_THREADS: int = 0                             # 0 = os.cpu_count()
    SLM_CTX: int = 4096
    SLM_USE_MMAP: bool = True                        # False = read the whole GGUF into RAM at startup (slow disks)
    SLM_TEMPERATURE: float = 0.2
    SLM_MAX_TOKENS_ANSWER: int = 520
    SLM_MAX_TOKENS_INTENT: int = 260
    SLM_MAX_TOKENS_EXPLAIN: int = 260
    SLM_TOP_P: float = 0.9
    SLM_REPETITION_PENALTY: float = 1.15             # discourages small-model loops ("X; X; X; ...")

    # Data sources (all free/open)
    CONTACT_EMAIL: str = "aaronr43210@gmail.com"   # Nominatim usage policy requires a contact
    NOMINATIM_URL: str = "https://nominatim.openstreetmap.org"
    GEOCODE_COUNTRY_CODES: str = "in"
    OVERPASS_MIRRORS: str = (
        "https://overpass-api.de/api/interpreter,"
        "https://maps.mail.ru/osm/tools/overpass/api/interpreter,"
        "https://overpass.kumi.systems/api/interpreter,"
        "https://overpass.private.coffee/api/interpreter"
    )
    OVERPASS_TIMEOUT_S: float = 15.0
    OVERPASS_BREAKER_S: int = 900
    WEATHER_API_URL: str = "https://api.open-meteo.com/v1"
    AIR_QUALITY_API_URL: str = "https://air-quality-api.open-meteo.com/v1"
    OSRM_URL: str = "https://router.project-osrm.org"
    WAQI_URL: str = "https://api.waqi.info"
    WAQI_TOKEN: str = ""

    # Cache TTLs (seconds)
    TTL_GEOCODE: int = 30 * 86400
    TTL_NEGATIVE: int = 600            # "not found" answers are retried after 10 min, not cached for a month
    TTL_OSM: int = 24 * 3600
    TTL_WEATHER: int = 1800
    TTL_AIR: int = 3600
    TTL_ROUTE: int = 6 * 3600
    TTL_WAQI: int = 1800

    # Sessions (conversation memory) and health probe
    SESSION_TTL_S: int = 1800
    SESSION_MAX: int = 2000
    HEALTH_PROBE_LAT: float = 0.0     # any point works; probes only check that the upstream answers
    HEALTH_PROBE_LON: float = 0.0

    # Search defaults (used only when the user gives none, and always reported)
    DEFAULT_RADIUS_KM: float = 10.0
    MAX_RADIUS_KM: float = 50.0
    DEFAULT_MAX_TRAVEL_MIN: int = 45

    @property
    def overpass_mirrors(self) -> List[str]:
        return [m.strip() for m in self.OVERPASS_MIRRORS.split(",") if m.strip()]


@lru_cache()
def get_settings() -> Settings:
    return Settings()
