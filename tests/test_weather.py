"""Weather over any background, snow that settles, and the empty skies."""

import pytest
import torch

import gui
import settings
import shapes3d
import weather

H, W = 108, 192


def person(left=0.3, right=0.7, top=0.4):
    """A matte with a block of a person in it: (1, 1, H, W)."""
    a = torch.zeros(1, 1, H, W)
    a[:, :, int(top * H):, int(left * W):int(right * W)] = 1.0
    return a


def test_the_panel_lists_every_kind():
    assert [name for name, _ in gui.WEATHER] == list(weather.KINDS)


def test_the_saved_defaults_know_the_weather():
    assert settings.clean({"weather": "rain", "settle": False, "bogus": 1}) == \
        {"weather": "rain", "settle": False}


@pytest.mark.parametrize("kind", list(weather.KINDS))
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_every_kind_draws_both_layers_on_any_frame(kind, dtype):
    w = weather.Weather(kind, "cpu")
    frame = torch.full((1, 3, H, W), 0.2, dtype=dtype)
    seen = 0.0
    for _ in range(10):
        w.step(1 / 30, H, W)
        for near in (False, True):
            out = w.layer(frame, near)
            assert out.shape == frame.shape and out.dtype == dtype
            assert float(out.min()) >= 0 and float(out.max()) <= 1.0
            seen += float((out.float() - 0.2).abs().sum())
    assert seen > 0


def test_flakes_keep_their_own_colour_where_they_pile_up():
    w = weather.Weather("leaves", "cpu")
    w.flakes.pos[:] = torch.tensor([0.0, 0.0, 6.0])       # all of them on one spot
    out = w.layer(torch.zeros(1, 3, H, W), near=True)[0]
    brightest = out[:, H // 2, W // 2]
    # Leaves are orange and brown: red over green over blue, never white.
    assert brightest[0] > brightest[1] > brightest[2]
    assert float(out.max()) <= 1.0


def test_snow_settles_on_top_of_you_and_only_there():
    s = weather.Settled()
    a = person()
    for _ in range(30 * 20):
        s.update(a, 1 / 30)
    out = s.draw(torch.zeros(1, 3, H, W))[0, 0]
    top = int(0.4 * H)
    assert float(out[top - 1:top + 2, W // 2].max()) > 0.9       # along the shoulders
    assert float(out[H - 10:, :].max()) == 0                      # not down your front
    assert float(out[:top - 6, :].max()) == 0                     # not in the air
    assert float(out[top + 12:, int(0.3 * W) - 2:int(0.3 * W) + 2].max()) == 0   # nor your sides


def test_it_takes_a_while_to_show():
    s = weather.Settled()
    for _ in range(30):                       # one second
        s.update(person(), 1 / 30)
    assert float(s.draw(torch.zeros(1, 3, H, W)).max()) == 0


def test_moving_shakes_it_off():
    s = weather.Settled()
    for _ in range(30 * 20):
        s.update(person(), 1 / 30)
    assert float(s.draw(torch.zeros(1, 3, H, W)).max()) > 0.9
    for _ in range(30):                       # a second somewhere else entirely
        s.update(person(left=0.05, right=0.25, top=0.6), 1 / 30)
    assert float(s.draw(torch.zeros(1, 3, H, W)).max()) == 0


def test_no_one_in_shot_is_no_snow():
    s = weather.Settled()
    for _ in range(30 * 20):
        s.update(torch.zeros(1, 1, H, W), 1 / 30)
    assert float(s.draw(torch.zeros(1, 3, H, W)).max()) == 0


@pytest.mark.parametrize("name", ["snow", "stars", "aurora"])
def test_a_sky_renders_with_nothing_floating_in_it(name):
    bg = shapes3d.ShapesBackground(f"shapes:{name}", "cpu", torch.float32)
    assert len(bg.swarm.pos) == 0
    bg.observe(person())                      # the matte has nothing to push
    first = bg.sample(H, W)
    assert first.shape == (1, 3, H, W)
    assert 0 <= float(first.min()) and float(first.max()) <= 1


def test_stars_are_bright_points_that_twinkle():
    now = weather.stars(H, W, 0.0, "cpu")
    later = weather.stars(H, W, 0.7, "cpu")
    assert float(now.max()) > 0.5
    assert float((now > 0.05).float().mean()) < 0.1      # points, not a haze
    lit = now.sum(0) > 0.05
    assert bool(((now - later).abs().sum(0)[lit] > 0).any())
    assert bool(((now > 0) == (later > 0)).all())          # dimmer and brighter, not moved


def test_the_amount_goes_from_nothing_to_twice_as_much():
    def covered(amount):
        w = weather.Weather("snow", "cpu")
        w.flakes.amount = amount
        frame = torch.zeros(1, 3, H, W)
        return float(w.layer(w.layer(frame, False), True).sum())

    assert covered(0.0) == 0
    assert 0 < covered(0.5) < covered(1.0) < covered(2.0)


def test_more_snow_settles_sooner():
    def after_two_seconds(fall):
        s = weather.Settled()
        for _ in range(60):
            s.update(person(), 1 / 30, fall=fall)
        return float(s.amount.max())

    assert after_two_seconds(2.0) > after_two_seconds(1.0) > after_two_seconds(0.0) == 0


def test_the_aurora_follows_its_dial():
    bg = shapes3d.ShapesBackground("shapes:aurora", "cpu", torch.float32)
    bg.aurora = 0.0
    without = bg.render(H, W)
    bg.aurora = 2.0
    assert float((bg.render(H, W) - without).max()) > 0.1


def test_a_shooting_star_is_a_thin_faint_streak_that_burns_out():
    streak = weather.meteor(H * 4, W * 4, weather.METEOR / 2, 0.5, 0.2, 0.8, 0.6, "cpu")
    assert 0.3 < float(streak.max()) <= 0.6               # seen, and not a firework
    assert float((streak.sum(0) > 0.05).float().mean()) < 0.001
    assert weather.meteor(H, W, weather.METEOR, 0.5, 0.2, 0.8, 0.6, "cpu") is None


def test_only_a_night_sky_has_shooting_stars():
    assert not shapes3d.ShapesBackground("shapes:snow", "cpu", torch.float32).shoot()
    sky = shapes3d.ShapesBackground("shapes:stars", "cpu", torch.float32)
    quiet = sky.render(H, W)
    assert sky.shoot()
    sky.swarm.t += weather.METEOR / 2
    assert float((sky.render(H, W) - quiet).max()) > 0.2
    sky.swarm.t += weather.METEOR                          # and then it is gone again
    sky.render(H, W)
    assert sky._meteor is None


def test_one_comes_by_itself_now_and_then():
    def seconds_to_the_first(per_minute, limit=600):
        sky = shapes3d.ShapesBackground("shapes:stars", "cpu", torch.float32)
        sky.shooting_stars = per_minute
        for second in range(1, limit + 1):
            sky.swarm.t = float(second)
            sky.render(27, 48)
            if sky._meteor is not None:
                return second
        return None

    # As built, 0.6 a minute: one every 50 to 150 seconds.
    assert 50 <= seconds_to_the_first(0.6) <= 150
    assert seconds_to_the_first(6.0) <= 15
    assert seconds_to_the_first(0.0) is None               # and never, if you say so


def test_turning_the_rate_up_brings_the_next_one_sooner():
    sky = shapes3d.ShapesBackground("shapes:stars", "cpu", torch.float32)
    sky.swarm.t = 20.0
    sky.render(27, 48)
    assert sky._meteor is None
    sky.shooting_stars = 6.0                               # the wait is not already fixed
    sky.swarm.t = 35.0
    sky.render(27, 48)
    assert sky._meteor is not None
