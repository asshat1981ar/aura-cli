def test_typer_importable():
    import typer
    assert hasattr(typer, "Typer")

def test_autogen_importable():
    import autogen
    assert hasattr(autogen, "AssistantAgent")
