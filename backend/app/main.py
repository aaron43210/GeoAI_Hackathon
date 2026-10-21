import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.agent.pipeline import Pipeline
from app.agent.session import SessionStore
from app.api import activities, ask, health
from app.config import get_settings
from app.data_services.air_quality_service import AirQualityService, WAQIService
from app.data_services.cache import InMemoryCache
from app.data_services.geocoding_service import GeocodingService
from app.data_services.http import HostPolicy, HttpGateway
from app.data_services.osm_service import OSMService
from app.data_services.routing_service import RoutingService
from app.data_services.weather_service import WeatherService
from app.slm.runtime import SLMRuntime

settings = get_settings()
logging.basicConfig(level=settings.LOG_LEVEL, format="%(asctime)s %(name)s %(levelname)s %(message)s", force=True)
logger = logging.getLogger("terrapulse")

# Per-host etiquette: Nominatim ≤1 req/s (usage policy), OSRM demo ≤1 req/s, Overpass ≤2 concurrent slots.
HOST_POLICIES = {
    "nominatim.openstreetmap.org": HostPolicy(min_interval_s=1.05, max_concurrent=1),
    "router.project-osrm.org": HostPolicy(min_interval_s=1.0, max_concurrent=1),
    "overpass-api.de": HostPolicy(max_concurrent=2),
    "maps.mail.ru": HostPolicy(max_concurrent=2),
    "overpass.kumi.systems": HostPolicy(max_concurrent=2),
    "overpass.private.coffee": HostPolicy(max_concurrent=2),
    "api.open-meteo.com": HostPolicy(min_interval_s=0.2, max_concurrent=4),
    "air-quality-api.open-meteo.com": HostPolicy(min_interval_s=0.2, max_concurrent=4),
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting %s v%s", settings.API_TITLE, settings.API_VERSION)
    http = HttpGateway(user_agent=f"TerraPulseAI/{settings.API_VERSION} (GeoAI hackathon; {settings.CONTACT_EMAIL})",
                       policies=HOST_POLICIES)
    cache = InMemoryCache()
    slm = getattr(app.state, "preloaded_slm", None) or SLMRuntime()   # a host may load the model before startup
    if not slm.ready:
        await slm.load()
    app.state.http, app.state.cache, app.state.slm = http, cache, slm
    app.state.pipeline = Pipeline(
        geocoder=GeocodingService(http, cache), osm=OSMService(http, cache), weather=WeatherService(http, cache),
        air=AirQualityService(http, cache), waqi=WAQIService(http, cache), routing=RoutingService(http, cache),
        slm=slm, sessions=SessionStore())
    yield
    await http.close()
    logger.info("Shut down")


app = FastAPI(title=settings.API_TITLE, version=settings.API_VERSION, lifespan=lifespan,
              description="Outdoor activity assistant: best nearby place and time, grounded in live open data, "
                          "written by a ≤3B open small language model.")
app.add_middleware(CORSMiddleware, allow_origins=[o.strip() for o in settings.CORS_ORIGINS.split(",")],
                   allow_methods=["*"], allow_headers=["*"])
app.include_router(ask.router, prefix="/api/v1", tags=["Ask"])
app.include_router(health.router, prefix="/api/v1", tags=["Health"])
app.include_router(activities.router, prefix="/api/v1", tags=["Activities"])

# Serve the built frontend from the same origin when present (single URL for judges)
_dist = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if not _dist.exists():
    _dist = Path(__file__).resolve().parents[1] / "static"
if _dist.exists():
    if (_dist / "assets").exists():
        app.mount("/assets", StaticFiles(directory=_dist / "assets"), name="assets")

    @app.get("/", include_in_schema=False)
    async def index():
        return FileResponse(_dist / "index.html")
else:
    @app.get("/", include_in_schema=False)
    async def root():
        return {"service": settings.API_TITLE, "version": settings.API_VERSION, "docs": "/docs",
                "ask": "/api/v1/ask?q=..."}
