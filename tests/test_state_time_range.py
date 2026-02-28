import pytest
from pixiu.state import State

def test_state_has_time_range_fields():
    state = State()
    assert hasattr(state, 'time_range_mode')
    assert hasattr(state, 'quick_range')
    assert hasattr(state, 'year_range')
    assert hasattr(state, 'backtest_start_date')
    assert hasattr(state, 'backtest_end_date')

def test_set_quick_range_updates_dates():
    state = State()
    state.set_quick_range("12m")
    assert state.time_range_mode == "quick"
    assert state.quick_range == "12m"
    assert state.backtest_start_date != ""
    assert state.backtest_end_date != ""

def test_set_year_range_this_year():
    state = State()
    state.set_year_range("this_year")
    assert state.time_range_mode == "year"
    assert state.year_range == "this_year"
