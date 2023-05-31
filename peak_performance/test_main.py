def test_version():
    import peak_performance as pp

    assert pp.__version__.count(".") == 2
