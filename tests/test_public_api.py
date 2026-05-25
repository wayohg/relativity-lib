"""Basic public API checks for the cleaned __init__.py files."""


def test_top_level_imports():
    import relativity

    assert hasattr(relativity, "C")
    assert hasattr(relativity, "Particle")
    assert hasattr(relativity, "sr")
    assert hasattr(relativity, "plotting")


def test_sr_namespaces_do_not_flatten_common_functions():
    from relativity import sr

    assert hasattr(sr, "kinematics")
    assert hasattr(sr, "dynamics")
    assert hasattr(sr.kinematics, "gamma")
    assert not hasattr(sr, "gamma")


def test_plotting_namespaces_available():
    from relativity import plotting

    assert hasattr(plotting, "kinematics")
    assert hasattr(plotting, "spacetime")
    assert hasattr(plotting.kinematics, "plot_gamma_vs_beta")
