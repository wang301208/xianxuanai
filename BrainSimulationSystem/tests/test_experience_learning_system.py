from BrainSimulationSystem.learning.experience import ExperienceLearningSystem


def test_experience_learning_generates_non_zero_action_values():
    learner = ExperienceLearningSystem()
    learner.exploration_rate = 0.0

    state = {'signal': 1.0}
    next_state = {'signal': 1.0}
    available_actions = [0, 1]

    # Inject synthetic training data favoring action 1
    for _ in range(40):
        learner.store_experience(
            state,
            action=1,
            reward=1.0,
            next_state=next_state,
            available_actions=available_actions
        )

    result = learner.learn_from_experience(batch_size=32)
    assert result['status'] == 'success'

    state_key = learner._state_to_key(state)
    learned_value = learner.value_network.get(f"{state_key}_1", 0.0)
    assert learned_value > 0.0

    recommended_action = learner.get_action_recommendation(state, available_actions)
    assert recommended_action == 1
