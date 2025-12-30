from governance.queue import ProposalQueue


def test_enqueue_and_approve(tmp_path):
    called = {"reloaded": False}

    def _reload():
        called["reloaded"] = True

    proposal_dir = tmp_path / "proposals"
    blueprint_dir = tmp_path / "blueprints"
    queue = ProposalQueue(
        proposal_dir=proposal_dir,
        blueprint_dir=blueprint_dir,
        reload_callback=_reload,
    )

    data = {
        "role_name": "Tester",
        "core_prompt": "Perform tests",
        "authorized_tools": [],
        "subscribed_topics": [],
    }

    proposal = queue.enqueue(data)
    assert proposal.exists()
    assert proposal in queue.list_pending()

    result = queue.approve(proposal, commit=False)
    assert result.exists()
    assert called["reloaded"] is True
    assert not proposal.exists()
