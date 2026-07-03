from evaluation.run_hybrid_eval import is_single_attribute, rrf


def test_rrf_single_list_preserves_order():
    assert rrf(["a", "b", "c"]) == ["a", "b", "c"]


def test_rrf_item_in_both_lists_beats_item_in_one():
    # "b" is ranked in both lists; "a" and "c" appear in only one each.
    fused = rrf(["a", "b"], ["b", "c"])
    assert fused[0] == "b"


def test_rrf_items_unique_to_one_list_still_included():
    fused = rrf(["a", "b"], ["c"])
    assert set(fused) == {"a", "b", "c"}


def test_single_attribute_price_only():
    assert is_single_attribute("hotels under 20000 LKR", "Colombo") is True


def test_single_attribute_rating_only():
    assert is_single_attribute("top-rated hotels", "Colombo") is True


def test_multi_dimensional_is_not_single_attribute():
    q = "4 star hotels under 20000 with rating above 4"
    assert is_single_attribute(q, "Colombo") is False


def test_accessibility_is_not_single_attribute():
    assert is_single_attribute("hotels with easy access avoiding traffic", "Colombo") is False
