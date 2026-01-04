import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "backend/monitoring")))

from global_workspace import GlobalWorkspace, WorkspaceMessage


class DummyModule:
    def __init__(self) -> None:
        self.received = []

    def receive_broadcast(self, sender, state, attention):
        self.received.append((sender, state, attention))


def test_broadcast_propagates_state() -> None:
    gw = GlobalWorkspace()
    a = DummyModule()
    b = DummyModule()
    gw.register_module("a", a)
    gw.register_module("b", b)
    gw.broadcast("a", {"value": 1}, [0.5, 0.5])
    assert b.received == [("a", {"value": 1}, [0.5, 0.5])]
    assert gw.state("a") == {"value": 1}
    assert gw.attention("a") == [0.5, 0.5]


def test_subscribe_state_receives_updates() -> None:
    gw = GlobalWorkspace()
    received = []
    gw.subscribe_state("self_model", lambda s: received.append(s))
    gw.broadcast("self_model", {"agent": "a", "summary": "hi"})
    assert received == [{"agent": "a", "summary": "hi"}]


def test_cross_attention_fuses_states() -> None:
    gw = GlobalWorkspace()
    text = DummyModule()
    vision = DummyModule()
    obs = DummyModule()
    gw.register_module("text", text)
    gw.register_module("vision", vision)
    gw.register_module("obs", obs)

    def fuse(t_state, v_state, t_att, v_att):
        fused = {"text": t_state["t"], "vision": v_state["v"]}
        return fused, [0.5, 0.5]

    gw.register_cross_attention("text", "vision", fuse)

    gw.broadcast("text", {"t": "hello"}, [1.0, 0.0])
    obs.received.clear()
    gw.broadcast("vision", {"v": "img"}, [0.0, 1.0])

    assert any(
        msg[0] == "text|vision" and msg[1] == {"text": "hello", "vision": "img"}
        for msg in obs.received
    )


def test_sparse_attention_routes_to_high_attention_modules() -> None:
    gw = GlobalWorkspace()
    a = DummyModule()
    b = DummyModule()
    c = DummyModule()
    gw.register_module("a", a)
    gw.register_module("b", b)
    gw.register_module("c", c)
    gw.broadcast("b", {}, [0.9])
    gw.broadcast("c", {}, [0.1])
    b.received.clear()
    c.received.clear()
    gw.broadcast("a", {"data": 1}, [0.5], strategy="sparse", k=1)
    assert b.received == [("a", {"data": 1}, [0.5])]
    assert c.received == []


def test_local_attention_targets_specified_modules() -> None:
    gw = GlobalWorkspace()
    a = DummyModule()
    b = DummyModule()
    c = DummyModule()
    gw.register_module("a", a)
    gw.register_module("b", b)
    gw.register_module("c", c)
    gw.broadcast("a", {"data": 2}, [0.3], strategy="local", targets=["b"])
    assert b.received == [("a", {"data": 2}, [0.3])]
    assert c.received == []


def test_publish_message_propagate_uses_importance_as_attention_when_missing() -> None:
    gw = GlobalWorkspace()
    receiver = DummyModule()
    gw.register_module("receiver", receiver)
    gw.set_attention_threshold(0.2)

    gw.publish_message(
        WorkspaceMessage(type="test", source="unit", payload={"x": 1}, importance=0.1),
        propagate=True,
    )
    assert receiver.received == []

    gw.publish_message(
        WorkspaceMessage(type="test", source="unit", payload={"x": 2}, importance=0.6),
        propagate=True,
    )
    assert receiver.received
    sender, state, attention = receiver.received[-1]
    assert sender.startswith("blackboard:")
    assert state.get("payload", {}).get("x") == 2
    assert attention == [0.6]
