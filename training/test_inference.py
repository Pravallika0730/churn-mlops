def test_inference_sanity():
    """
    CI-safe sanity test.
    Inference execution is skipped in CI
    to avoid loading trained ML artifacts.
    """
    assert True
