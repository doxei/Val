"""Connecteurs n8n : chaque workflow déclaré devient un outil, appelé par webhook."""
import json

import httpx
import pytest

from valdar.config.loader import Identity, N8nConfig, PermissionsConfig
from valdar.tools.n8n import add_n8n_tools, tool_name
from valdar.tools.registry import Registry, decide


def _cfg(**wf):
    return N8nConfig(enabled=True, workflows=wf or {
        "notifier_telephone": {"description": "Envoie une notification au téléphone.",
                               "params": {"message": "le texte"}, "required": ["message"],
                               "tier": "safe"},
        "envoyer_mail": {"description": "Envoie un mail.", "params": {"a": "dest", "texte": "x"},
                         "tier": "elevated", "confirm": True},
    })


def _client(seen):
    def handler(req: httpx.Request) -> httpx.Response:
        seen.append((req.url.path, json.loads(req.content)))
        if req.url.path.endswith("/panne"):
            return httpx.Response(500, text="boum")
        return httpx.Response(200, json={"ok": True, "envoye": "oui"})
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_each_workflow_becomes_a_tool_that_calls_its_webhook():
    seen = []
    reg = Registry()
    names = add_n8n_tools(reg, _cfg(), _client(seen))
    assert names == ["n8n_notifier_telephone", "n8n_envoyer_mail"]
    tool = reg.get("n8n_notifier_telephone")
    assert tool.schema()["function"]["parameters"]["required"] == ["message"]
    out = tool.fn(message="l'impression est finie")
    assert seen == [("/webhook/notifier_telephone", {"message": "l'impression est finie"})]
    assert '"envoye": "oui"' in out


def test_permissions_follow_the_declared_tier():
    reg = Registry()
    add_n8n_tools(reg, _cfg(), _client([]))
    perms = PermissionsConfig()
    guest = Identity(person="zoe", name="Zoé", role="guest", confidence=0.95)
    owner = Identity(person="olivier", name="Olivier", role="owner", confidence=0.95)
    assert decide(reg.get("n8n_notifier_telephone"), guest, perms).allowed
    assert not decide(reg.get("n8n_envoyer_mail"), guest, perms).allowed
    d = decide(reg.get("n8n_envoyer_mail"), owner, perms)
    assert d.allowed and d.needs_confirmation


def test_errors_are_told_not_raised():
    reg = Registry()
    add_n8n_tools(reg, _cfg(panne={"description": "x", "tier": "safe"}), _client([]))
    assert "a échoué (500)" in reg.get("n8n_panne").fn()
    down = N8nConfig(url="http://127.0.0.1:9", timeout_seconds=0.5,
                     workflows={"x": {"description": "x", "tier": "safe"}})
    reg2 = Registry()
    add_n8n_tools(reg2, down)
    assert "injoignable" in reg2.get("n8n_x").fn()


def test_config_is_checked():
    with pytest.raises(ValueError):
        N8nConfig(workflows={"x": {"description": "x", "tier": "admin"}})
    with pytest.raises(ValueError):
        N8nConfig(workflows={"x": {"description": "x", "required": ["y"]}})
    assert tool_name("Allumer la Lumière!") == "n8n_allumer_la_lumi_re"
