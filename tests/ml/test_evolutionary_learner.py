from backend.ml.evolutionary_learner import (
    EvolutionaryLearner,
    EvolutionaryPopulation,
    SearchDimension,
    GAConfig,
)


def paraboloid(params):
    x = params["x"] - 3.0
    y = params["y"] + 2.0
    return 1.0 / (1.0 + x * x + y * y)


def test_evolutionary_learner_finds_peak():
    space = [
        SearchDimension("x", -10.0, 10.0),
        SearchDimension("y", -10.0, 10.0),
    ]
    learner = EvolutionaryLearner(
        search_space=space,
        evaluator=paraboloid,
        ga_config=GAConfig(population_size=24, mutation_sigma=0.3),
    )
    best_params, best_fitness = learner.run(generations=15)

    assert abs(best_params["x"] - 3.0) < 1.0
    assert abs(best_params["y"] + 2.0) < 1.0
    assert best_fitness > 0.4


def test_population_callbacks_receive_records():
    space = [SearchDimension("x", -1.0, 1.0)]
    history = []

    def callback(record):
        history.append(record)

    population = EvolutionaryPopulation(
        search_space=space,
        evaluator=lambda p: 1.0 - abs(p["x"]),
        ga_config=GAConfig(population_size=6, mutation_sigma=0.2),
        callbacks=[callback],
    )
    result = population.run(generations=4)

    assert history, "Callbacks should receive evaluation records"
    assert result.best_fitness > 0.0
