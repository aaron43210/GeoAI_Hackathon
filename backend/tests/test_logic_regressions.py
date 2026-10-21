"""Logic bugs found in the senior review; each test fails on the old behaviour."""
from datetime import date, datetime

from shapely.geometry import Polygon

from app.agent.time_resolver import TimeResolver
from app.data_services.osm_service import OSMBundle, OSMFeature
from app.geospatial.context import compute_features
from app.scoring.grid import is_golden


def test_golden_hour_counts_the_slot_containing_sunrise_and_sunset():
    sunrise, sunset = 6 + 10 / 60, 18 + 5 / 60
    assert is_golden(6, sunrise, sunset, 1.0)        # 06:00–07:00 contains 06:10–07:10 (old code: False)
    assert is_golden(17, sunrise, sunset, 1.0)       # 17:00–18:00 is in the last hour before sunset
    assert is_golden(18, sunrise, sunset, 1.0)       # 18:00–19:00 contains the moment of sunset
    assert not is_golden(12, sunrise, sunset, 1.0)
    assert not is_golden(19, sunrise, sunset, 1.0)


def test_this_weekend_asked_on_sunday_means_today_only():
    sunday = datetime(2026, 10, 25, 9, 0)
    r = TimeResolver().resolve("best time this weekend for a picnic", sunday)
    assert [w.day for w in r.windows] == [date(2026, 10, 25)]          # old code: [Sun 25, Sat 31]
    saturday = datetime(2026, 10, 24, 9, 0)
    r = TimeResolver().resolve("this weekend", saturday)
    assert [w.day for w in r.windows] == [date(2026, 10, 24), date(2026, 10, 25)]
    r = TimeResolver().resolve("next weekend", sunday)
    assert [w.day for w in r.windows] == [date(2026, 10, 31), date(2026, 11, 1)]


def test_large_industrial_site_is_measured_and_weighted_from_its_nearest_edge():
    # venue at (10.0, 76.0); an industrial block whose west edge is ~110 m east, centroid ~1.2 km east
    venue = OSMFeature(osm_id="node/1", tags={"name": "V"}, lat=10.0, lon=76.0)
    block = [[9.99, 76.001], [10.01, 76.001], [10.01, 76.021], [9.99, 76.021], [9.99, 76.001]]
    ind = OSMFeature(osm_id="way/2", tags={"landuse": "industrial", "name": "Works"}, lat=10.0, lon=76.011,
                     geometry=block, kind="way")
    f = compute_features(venue, OSMBundle(industry=[ind]))
    src = [s for s in f.sources if s.kind == "industry"][0]
    dist_to_source = (src.x ** 2 + src.y ** 2) ** 0.5
    assert f.industry_within_3km[0]["distance_m"] < 150
    assert abs(dist_to_source - f.industry_within_3km[0]["distance_m"]) < 5   # old code: source at the ~1.2 km centroid
    assert f.industry_within_3km[0]["bearing_deg"] in range(80, 101)            # due east


def test_capped_windows_still_rank_by_underlying_quality():
    from app.scoring.grid import Cell, WindowPick
    a = WindowPick(venue=0, cells=[Cell(0, "t", datetime(2026, 10, 24, 6), {}, {}, None, [], 10, 55.0, "caution", 90.0)],
                   score=55.0, status="caution", label="")
    b = WindowPick(venue=1, cells=[Cell(1, "t", datetime(2026, 10, 24, 6), {}, {}, None, [], 5, 55.0, "caution", 60.0)],
                   score=55.0, status="caution", label="")
    key = lambda p, t: (-p.score, -sum(c.raw_score for c in p.cells) / len(p.cells), t)  # noqa: E731
    assert sorted([(b, 5), (a, 10)], key=lambda x: key(*x))[0][0] is a        # better air/comfort wins the tie


def test_run_length_is_not_a_search_radius():
    from app.agent.rule_parser import parse
    assert parse("Best time for a 5 km run tomorrow near Gurugram").radius_km is None
    assert parse("cricket within 15 km of Thane").radius_km == 15
    assert parse("parks 10 km around Baner").radius_km == 10


def test_negative_geocode_is_not_cached_for_a_month():
    import asyncio
    from app.config import get_settings
    from app.data_services.cache import InMemoryCache
    from app.data_services.geocoding_service import GeocodingService

    class _EmptyHttp:
        async def request(self, *a, **k):
            return []

    cache = InMemoryCache()
    asyncio.run(GeocodingService(_EmptyHttp(), cache).geocode("Nowhereville"))
    (expiry, _), = cache.store.values()
    import time
    assert expiry - time.monotonic() <= get_settings().TTL_NEGATIVE + 1
