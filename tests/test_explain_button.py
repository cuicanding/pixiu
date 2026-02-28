"""Tests for explain button components."""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock


class TestExplainPrompts:
    """Tests for explanation prompt templates."""

    def test_get_prompt_total_return(self):
        from pixiu.services.explain_prompts import get_prompt, EXPLAIN_PROMPTS
        
        result = get_prompt("total_return", value="25.5%")
        assert "总收益率" in result
        assert "25.5%" in result

    def test_get_prompt_sharpe_ratio(self):
        from pixiu.services.explain_prompts import get_prompt
        
        result = get_prompt("sharpe_ratio", value="1.5")
        assert "夏普比率" in result
        assert "1.5" in result

    def test_get_prompt_unknown_concept(self):
        from pixiu.services.explain_prompts import get_prompt
        
        result = get_prompt("unknown_concept", value="test")
        assert "unknown_concept" in result

    def test_get_prompt_with_regime(self):
        from pixiu.services.explain_prompts import get_prompt
        
        result = get_prompt("regime_trend", regime="趋势行情")
        assert "趋势行情" in result

    def test_all_prompts_have_required_keys(self):
        from pixiu.services.explain_prompts import EXPLAIN_PROMPTS
        
        expected_keys = [
            "total_return", "annualized_return", "sharpe_ratio",
            "max_drawdown", "win_rate", "profit_loss_ratio"
        ]
        for key in expected_keys:
            assert key in EXPLAIN_PROMPTS, f"Missing prompt for {key}"


class TestExplainState:
    """Tests for State explanation methods."""

    def test_state_has_explain_fields(self):
        from pixiu.state import State
        
        state = State()
        assert hasattr(state, "explain_modal_open")
        assert hasattr(state, "current_explanation")
        assert hasattr(state, "ai_explaining")
        assert state.explain_modal_open == False
        assert state.current_explanation == ""
        assert state.ai_explaining == False

    def test_close_explain_modal(self):
        from pixiu.state import State
        
        state = State()
        state.explain_modal_open = True
        state.close_explain_modal()
        assert state.explain_modal_open == False

    @pytest.mark.asyncio
    async def test_explain_concept_opens_modal(self):
        from pixiu.state import State
        
        state = State()
        state.glm_api_key = "test_key"
        
        with patch("pixiu.services.ai_service.ZhipuAI") as mock_zhipu:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Test explanation"
            mock_client.chat.completions.create = MagicMock(return_value=mock_response)
            mock_zhipu.return_value = mock_client
            
            gen = state.explain_concept("sharpe_ratio", "1.5")
            await gen.__anext__()
            assert state.explain_modal_open == True
            assert state.ai_explaining == True
            
            await gen.__anext__()
            assert state.ai_explaining == False
            assert state.current_explanation == "Test explanation"

    @pytest.mark.asyncio
    async def test_explain_concept_handles_error(self):
        from pixiu.state import State
        
        state = State()
        state.glm_api_key = "test_key"
        
        with patch("pixiu.services.ai_service.ZhipuAI") as mock_zhipu:
            mock_client = MagicMock()
            mock_client.chat.completions.create = MagicMock(side_effect=Exception("API Error"))
            mock_zhipu.return_value = mock_client
            
            gen = state.explain_concept("sharpe_ratio", "1.5")
            await gen.__anext__()
            await gen.__anext__()
            
            assert "失败" in state.current_explanation
            assert state.ai_explaining == False


class TestExplainComponents:
    """Tests for explain button components."""

    def test_explain_button_imports(self):
        from pixiu.components.explain_button import explain_button
        assert callable(explain_button)

    def test_metric_with_explain_imports(self):
        from pixiu.components.explain_button import metric_with_explain
        assert callable(metric_with_explain)

    def test_explain_modal_imports(self):
        from pixiu.components.explain_button import explain_modal
        assert callable(explain_modal)
