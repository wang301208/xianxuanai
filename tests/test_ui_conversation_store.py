import asyncio

from backend.forge.forge.ui_api import ConversationStore, ConversationTurn


def test_conversation_store_records_and_filters_by_session():
    store = ConversationStore(max_turns=10, max_session_turns=10, persist_path=None)

    async def run():
        await store.append(
            ConversationTurn(
                id="1",
                role="user",
                message="hello",
                timestamp="t",
                session_id="s1",
                attachments={"image": {"base64": "abc", "mime": "image/png"}},
            )
        )
        await store.append(
            ConversationTurn(
                id="2",
                role="agent",
                message="world",
                timestamp="t",
                session_id="s1",
            )
        )
        await store.append(
            ConversationTurn(
                id="3",
                role="user",
                message="other",
                timestamp="t",
                session_id="s2",
            )
        )
        all_hist = await store.history(limit=10)
        s1_hist = await store.history(limit=10, session_id="s1")
        return all_hist, s1_hist

    all_hist, s1_hist = asyncio.run(run())
    assert len(all_hist) == 3
    assert len(s1_hist) == 2
    assert s1_hist[0]["role"] == "agent"
    assert s1_hist[1]["attachments"]["image"]["mime"] == "image/png"

