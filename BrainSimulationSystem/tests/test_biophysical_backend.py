# -*- coding: utf-8 -*-

from __future__ import annotations

from BrainSimulationSystem.core.backends import get_backend


def test_biophysical_backend_builds_network_and_spikes():
    backend = get_backend("biophysical")
    config = {
        "simulation": {
            "backend": "biophysical",
            "dt": 1.0,
            "biophysical": {
                "seed": 1,
                "neurons_per_region": 20,
                "baseline_current_mean": 0.0,
                "baseline_current_std": 0.0,
                "noise_std": 0.0,
            },
        }
    }

    network = backend.build_network(config)
    assert hasattr(network, "step")

    # Provide per-region drive to reliably trigger spiking.
    network.set_input([25.0] * getattr(network, "region_count", 8))

    total_spikes = 0
    last_state = None
    for _ in range(25):
        last_state = network.step(1.0)
        assert isinstance(last_state, dict)
        total_spikes += int(last_state.get("spike_count", len(last_state.get("spikes", []))))

    assert last_state is not None
    assert "voltages" in last_state
    assert total_spikes > 0


def test_biophysical_stdp_updates_weights_when_enabled():
    backend = get_backend("biophysical")
    config = {
        "simulation": {
            "backend": "biophysical",
            "dt": 1.0,
            "biophysical": {
                "seed": 2,
                "neurons_per_region": 25,
                "baseline_current_mean": 0.0,
                "baseline_current_std": 0.0,
                "noise_std": 0.0,
                "stdp_enabled": True,
                "stdp_A_plus": 0.02,
                "stdp_A_minus": 0.024,
            },
        }
    }

    network = backend.build_network(config)
    network.set_input([28.0] * getattr(network, "region_count", 8))
    if hasattr(network, "set_neuromodulators"):
        network.set_neuromodulators({"dopamine": 2.0})

    network.step(1.0)  # build synapses + run a step
    w0 = getattr(network, "_syn_weight").copy()

    for _ in range(10):
        network.step(1.0)

    w1 = getattr(network, "_syn_weight")
    assert w1.size == w0.size and w1.size > 0
    assert (abs(w1 - w0) > 1e-6).any()


def test_biophysical_hh_mode_runs_and_spikes():
    backend = get_backend("biophysical")
    config = {
        "simulation": {
            "backend": "biophysical",
            "dt": 0.1,
            "biophysical": {
                "seed": 3,
                "neurons_per_region": 5,
                "neuron_model": "hh",
                "intra_connection_prob": 0.0,
                "inter_connection_prob": 0.0,
                "baseline_current_mean": 10.0,
                "baseline_current_std": 0.0,
                "noise_std": 0.0,
                "max_delay_ms": 5.0,
            },
        }
    }

    network = backend.build_network(config)
    network.set_input([0.0] * getattr(network, "region_count", 8))

    total_spikes = 0
    for _ in range(300):  # 30ms at dt=0.1ms
        state = network.step(0.1)
        total_spikes += int(state.get("spike_count", len(state.get("spikes", []))))

    assert total_spikes > 0


def test_biophysical_multicompartment_mode_runs_and_spikes():
    backend = get_backend("biophysical")
    config = {
        "simulation": {
            "backend": "biophysical",
            "dt": 0.1,
            "biophysical": {
                "seed": 4,
                "neurons_per_region": 5,
                "neuron_model": "mc",
                "intra_connection_prob": 0.0,
                "inter_connection_prob": 0.0,
                "baseline_current_mean": 20.0,
                "baseline_current_std": 0.0,
                "noise_std": 0.0,
                "max_delay_ms": 5.0,
            },
        }
    }

    network = backend.build_network(config)
    network.set_input([0.0] * getattr(network, "region_count", 8))

    total_spikes = 0
    for _ in range(300):  # 30ms at dt=0.1ms
        state = network.step(0.1)
        total_spikes += int(state.get("spike_count", len(state.get("spikes", []))))

    assert total_spikes > 0


def test_biophysical_receptor_synapse_model_runs_and_spikes():
    backend = get_backend("biophysical")
    config = {
        "simulation": {
            "backend": "biophysical",
            "dt": 1.0,
            "biophysical": {
                "seed": 5,
                "neurons_per_region": 20,
                "baseline_current_mean": 0.0,
                "baseline_current_std": 0.0,
                "noise_std": 0.0,
                "synapse_model": "receptor",
            },
        }
    }

    network = backend.build_network(config)
    network.set_input([30.0] * getattr(network, "region_count", 8))

    total_spikes = 0
    for _ in range(10):
        state = network.step(1.0)
        total_spikes += int(state.get("spike_count", len(state.get("spikes", []))))

    assert total_spikes > 0


def test_biophysical_stp_release_depresses_over_repeated_spikes():
    backend = get_backend("biophysical")
    config = {
        "simulation": {
            "backend": "biophysical",
            "dt": 1.0,
            "biophysical": {
                "seed": 6,
                "neurons_per_region": 12,
                "baseline_current_mean": 0.0,
                "baseline_current_std": 0.0,
                "noise_std": 0.0,
                "stp_enabled": True,
                "stp_apply_to": "exc",
                "stp_U": 0.4,
                "stp_tau_rec_ms": 800.0,
                "stp_tau_facil_ms": 0.0,  # pure depression
            },
        }
    }

    network = backend.build_network(config)
    network.set_input([0.0] * getattr(network, "region_count", 8))
    network.step(1.0)  # build synapses without triggering spiking

    syn_is_inh = getattr(network, "_syn_is_inh")
    exc_syn = (~syn_is_inh).nonzero()[0]
    assert exc_syn.size > 0
    idx = int(exc_syn[0])

    scale1 = float(network._stp_release_scale([idx], time_ms=0.0)[0])
    scale2 = float(network._stp_release_scale([idx], time_ms=10.0)[0])
    scale3 = float(network._stp_release_scale([idx], time_ms=20.0)[0])

    assert abs(scale1 - 1.0) < 1e-4
    assert scale2 < scale1
    assert scale3 < scale2


def test_biophysical_cell_type_assignment_produces_layer_specific_types():
    backend = get_backend("biophysical")
    config = {
        "simulation": {
            "backend": "biophysical",
            "dt": 1.0,
            "biophysical": {
                "seed": 7,
                "neurons_per_region": 20,
                "baseline_current_mean": 0.0,
                "baseline_current_std": 0.0,
                "noise_std": 0.0,
                "cell_types_enabled": True,
            },
        }
    }

    network = backend.build_network(config)
    cell_types = set(getattr(network, "_neuron_cell_type").tolist())
    assert "thalamus_tc" in cell_types
    assert "cortex_ib" in cell_types
    assert float(getattr(network, "_d").min()) < 0.1
    assert float(getattr(network, "_d").max()) > 1.0


def test_biophysical_hybrid_adex_izh_runs_and_spikes():
    backend = get_backend("biophysical")
    config = {
        "simulation": {
            "backend": "biophysical",
            "dt": 1.0,
            "biophysical": {
                "seed": 10,
                "neurons_per_region": 20,
                "neuron_model": "hybrid",
                "baseline_current_mean": 0.0,
                "baseline_current_std": 0.0,
                "noise_std": 0.0,
                "cell_types_enabled": True,
            },
        }
    }

    network = backend.build_network(config)
    network.set_input([30.0] * getattr(network, "region_count", 8))

    total_spikes = 0
    for _ in range(20):
        state = network.step(1.0)
        total_spikes += int(state.get("spike_count", len(state.get("spikes", []))))

    assert total_spikes > 0
    assert hasattr(network, "_adex_w")
    assert float(getattr(network, "_adex_w").max()) > 0.0


def test_biophysical_mc_active_dendrite_emits_plateau_trace():
    backend = get_backend("biophysical")
    config = {
        "simulation": {
            "backend": "biophysical",
            "dt": 0.1,
            "biophysical": {
                "seed": 8,
                "neurons_per_region": 4,
                "neuron_model": "mc",
                "intra_connection_prob": 0.0,
                "inter_connection_prob": 0.0,
                "baseline_current_mean": 20.0,
                "baseline_current_std": 0.0,
                "noise_std": 0.0,
                "max_delay_ms": 5.0,
                "mc_dendrite_active": True,
                "mc_dend_spike_threshold_mV": -60.0,
                "mc_dend_plateau_tau_ms": 5.0,
            },
        }
    }

    network = backend.build_network(config)
    network.set_input([0.0] * getattr(network, "region_count", 8))

    plateau_nonzero = False
    for _ in range(80):  # 8ms at dt=0.1ms
        state = network.step(0.1)
        plateau = state.get("dendrite_plateau") or {}
        if any(float(v) > 0.0 for v in plateau.values()):
            plateau_nonzero = True
            break

    assert plateau_nonzero


def test_biophysical_physiology_outputs_when_enabled():
    backend = get_backend("biophysical")
    config = {
        "simulation": {
            "backend": "biophysical",
            "dt": 1.0,
            "biophysical": {
                "seed": 9,
                "neurons_per_region": 10,
                "baseline_current_mean": 0.0,
                "baseline_current_std": 0.0,
                "noise_std": 0.0,
                "physiology_enabled": True,
            },
        }
    }

    network = backend.build_network(config)
    network.set_input([25.0] * getattr(network, "region_count", 8))
    state = network.step(1.0)

    phys = state.get("physiology")
    assert isinstance(phys, dict) and phys
    first_region = getattr(network, "regions", list(phys.keys()))[0]
    assert "atp" in phys[first_region]
    assert "blood_flow" in phys[first_region]
    assert "glia" in phys[first_region]
    assert "rate_hz" in phys[first_region]
