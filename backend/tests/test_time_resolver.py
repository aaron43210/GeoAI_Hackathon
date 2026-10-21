from datetime import date, datetime

import pytest

from app.agent.time_resolver import TimeResolver

NOW = datetime(2026, 10, 21, 9, 5)       # Wednesday 09:05 local


@pytest.fixture(scope="module")
def tr():
    return TimeResolver()


def days(res):
    return [w.day for w in res.windows]


def test_tomorrow_evening(tr):
    r = tr.resolve("running tomorrow evening", NOW)
    assert days(r) == [date(2026, 10, 22)]
    assert r.windows[0].start_h == 16.5 and r.windows[0].end_h == 20.0


def test_weekend_compares_both_days(tr):
    r = tr.resolve("best time this weekend", NOW)
    assert days(r) == [date(2026, 10, 24), date(2026, 10, 25)]
    assert r.compare


def test_saturday_or_sunday(tr):
    r = tr.resolve("Is Saturday or Sunday better for a hike", NOW)
    assert set(days(r)) == {date(2026, 10, 24), date(2026, 10, 25)} and r.compare


def test_next_two_days_morning_trims_past(tr):
    r = tr.resolve("morning walk in the next two days", NOW)
    assert days(r) == [date(2026, 10, 21), date(2026, 10, 22)]
    assert r.windows[0].start_h > 9.0          # today's window starts after "now"


def test_explicit_hour_passed_rolls_to_tomorrow(tr):
    r = tr.resolve("cycle to work at 8 AM", NOW)
    assert days(r) == [date(2026, 10, 22)]
    assert r.explicit_hour == 8


def test_hinglish(tr):
    r = tr.resolve("kal subah jogging", NOW)
    assert days(r) == [date(2026, 10, 22)] and r.windows[0].label == "subah"


def test_sunset_is_anchor_not_evening(tr):
    r = tr.resolve("sunset photos this evening", NOW)
    assert r.windows[0].anchor == "sunset"
    w = tr.apply_anchors(r.windows, {date(2026, 10, 21): (6.5, 18.2)})
    assert w[0].start_h == pytest.approx(18.2 - 1.25) and w[0].end_h == pytest.approx(18.2 + 0.75)


def test_no_time_is_assumed_and_said(tr):
    r = tr.resolve("where can I play tennis", NOW)
    assert r.assumed and r.notes


def test_explicit_date(tr):
    r = tr.resolve("mehendi on 2 Nov afternoon", NOW)
    assert days(r) == [date(2026, 11, 2)]


def test_horizon(tr):
    r = tr.resolve("on 30 dec", NOW)
    assert "beyond_weather_horizon" in tr.horizon_check(r, NOW)


def test_right_now(tr):
    r = tr.resolve("go jogging right now", NOW)
    assert r.windows[0].day == NOW.date() and r.windows[0].start_h <= 9.5 and r.windows[0].label == "now"
