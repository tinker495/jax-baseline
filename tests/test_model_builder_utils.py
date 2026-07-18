import sys
from types import SimpleNamespace

from model_builder.utils import print_flax_model_summary, print_haiku_model_summary


def test_print_flax_model_summary_is_lazy_and_uses_module_tabulate(capsys):
    calls = []

    class Model:
        def tabulate(self, *args):
            calls.append(args)
            return "flax summary"

    model = Model()
    print_flax_model_summary(False, "key", (model, "input"))
    assert calls == []
    assert capsys.readouterr().out == ""

    print_flax_model_summary(True, "key", (model, "input"))
    assert calls == [("key", "input")]
    assert capsys.readouterr().out == "flax summary\n"


def test_print_haiku_model_summary_is_lazy_and_uses_experimental_tabulate(capsys, monkeypatch):
    calls = []

    def tabulate(model):
        def summarize(*inputs):
            calls.append((model, inputs))
            return f"{model} summary"

        return summarize

    monkeypatch.setitem(
        sys.modules,
        "haiku",
        SimpleNamespace(experimental=SimpleNamespace(tabulate=tabulate)),
    )

    print_haiku_model_summary(False, ("actor", "input"))
    assert calls == []
    assert capsys.readouterr().out == ""

    print_haiku_model_summary(True, ("actor", "input"))
    assert calls == [("actor", ("input",))]
    assert capsys.readouterr().out == "actor summary\n"
