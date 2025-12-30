import sys
import types
from unittest.mock import patch

if 'BrainSimulationSystem.personality.custom_traits' not in sys.modules:
    custom_traits = types.ModuleType('BrainSimulationSystem.personality.custom_traits')

    class PersonalityCustomizer:  # pragma: no cover - simple stub for tests
        def __init__(self):
            pass

    custom_traits.PersonalityCustomizer = PersonalityCustomizer
    sys.modules['BrainSimulationSystem.personality.custom_traits'] = custom_traits

if 'BrainSimulationSystem.learning.experience' not in sys.modules:
    experience_module = types.ModuleType('BrainSimulationSystem.learning.experience')

    class ExperienceLearningSystem:  # pragma: no cover - simple stub for tests
        def __init__(self, memory_capacity: int = 10000):
            self.exploration_rate = 0.1
            self.learning_rate = 0.05

        def get_action_recommendation(self, state, available_actions):
            return available_actions[0] if available_actions else 0

        def store_experience(self, *args, **kwargs):
            return None

        def learn_from_experience(self, *args, **kwargs):
            return {'status': 'skipped'}

    experience_module.ExperienceLearningSystem = ExperienceLearningSystem
    sys.modules['BrainSimulationSystem.learning.experience'] = experience_module

from BrainSimulationSystem.meta_decision.core import MetaDecisionSystem
from BrainSimulationSystem.decision.core import DecisionEngine
from BrainSimulationSystem.models.decision import DecisionProcess


def _simulate_decision(
    meta: MetaDecisionSystem,
    context: dict,
    options: list,
    selected: int,
    expected_reward: float,
    actual_reward: float,
    utility: float,
    time_spent: float,
) -> None:
    process_data = {
        'options': options,
        'context': context,
        'selected': selected,
        'confidence': 0.9,
        'time_spent': time_spent,
    }
    meta.monitor_decision_process(process_data)
    meta.evaluate_decision_quality(
        {
            'selected_option': options[selected],
            'expected_outcome': {'reward': expected_reward},
            'actual_outcome': {'reward': actual_reward},
            'utility': utility,
        }
    )
    meta.decision_history[-1]['time_spent'] = time_spent


def test_quality_metrics_reflect_history_trends():
    meta = MetaDecisionSystem()
    base_option = {'expected_value': 1.0, 'risk': 0.1, 'cost': 0.1}
    options = [base_option, {'expected_value': 0.4, 'risk': 0.5, 'cost': 0.2}]

    for idx in range(5):
        context = {
            'scenario': 'consistent',
            'emotional_factor': 0.4,
            'cognitive_factor': 0.7,
        }
        utility = 0.7 + 0.1 * idx
        _simulate_decision(
            meta,
            context,
            options,
            0,
            expected_reward=0.6 + 0.05 * idx,
            actual_reward=0.6 + 0.05 * idx,
            utility=utility,
            time_spent=0.1,
        )

    meta._update_quality_metrics()
    metrics = meta.quality_metrics

    assert metrics['consistency'] > 0.9
    assert metrics['adaptability'] > 0.55
    assert metrics['efficiency'] > 0.55
    assert metrics['rationality'] > 0.85


def test_adjustments_persist_and_apply_next_cycle():
    meta = MetaDecisionSystem()
    poor_option = {'expected_value': 0.2, 'risk': 0.6, 'cost': 0.1}
    better_option = {'expected_value': 0.5, 'risk': 0.2, 'cost': 0.1}
    options = [poor_option, better_option]

    for idx in range(5):
        context = {
            'scenario': f'volatile-{idx % 2}',
            'emotional_factor': 0.85,
            'cognitive_factor': 0.3,
        }
        utility = 0.2 - 0.02 * idx
        _simulate_decision(
            meta,
            context,
            options,
            0,
            expected_reward=0.6,
            actual_reward=0.2,
            utility=utility,
            time_spent=2.0 + idx,
        )

    meta._update_quality_metrics()
    suggestion = meta.suggest_adjustments()
    suggested = suggestion['suggested_adjustments']
    process_adjustments = suggested['process']
    engine_adjustments = suggested['engine']

    assert process_adjustments['learning_rate'] != 0
    assert process_adjustments['exploration_rate'] != 0
    assert engine_adjustments['heuristic_weights']

    decision_process = DecisionProcess(None, {'decision_type': 'softmax'})
    decision_process.set_meta_adjustment_provider(meta)

    original_learning = decision_process.params.get('learning_rate', 0.1)
    original_temp = decision_process.params.get('temperature', 1.0)

    decision_process.process({'options': ['a', 'b'], 'context': {'foo': 'bar'}})

    assert decision_process.params['learning_rate'] != original_learning
    assert decision_process.params['temperature'] != original_temp

    engine = DecisionEngine()
    engine.meta_decider = meta

    with patch.object(engine, 'update_weights', wraps=engine.update_weights) as mocked_update:
        engine.make_decision(options, {'expected': {'reward': 0.3}})
        mocked_update.assert_called()
        delta_args = mocked_update.call_args[0][0]
        assert any(abs(v) > 0 for v in delta_args.values())
