from fastapi import APIRouter

from app.knowledge import get_knowledge

router = APIRouter()


@router.get("/activities")
async def list_activities():
    """Supported activities and personal profiles, straight from the knowledge base."""
    kb = get_knowledge()
    return {
        "activities": [{"id": a.id, "name": a.label, "synonyms": a.synonyms, "daylight": a.daylight,
                        "exertion": a.exertion, "default_duration_min": a.default_duration_min,
                        "limits": a.limits, "weights": a.weights} for a in kb.activities.values()],
        "profiles": [{"id": p.id, "name": p.label, "limits": p.limits} for p in kb.profiles.values()],
    }
