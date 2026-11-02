"""parda/__main__.py: Levenshtein distance and the "is this verify finding a value we already cover?" check."""
import pytest

from parda.__main__ import _edits, _known


@pytest.mark.parametrize("a, b, d", [
    ("", "", 0), ("abc", "", 3), ("", "abc", 3), ("kitten", "sitting", 3), ("flaw", "lawn", 2),
    ("akiaiosfodnn7example", "akiai0sfodnn7example", 1), ("abcde1234f", "abcde1234f", 0), ("ab", "ba", 2),
])
def test_edits(a, b, d):
    assert _edits(a, b) == d
    assert _edits(b, a) == d


KNOWN = {"priya.sharma@example.com", "akiaiosfodnn7example", "abcde1234f", "9876543210"}


def test_known_exact():
    assert _known("abcde1234f", KNOWN)
    assert _known("9876543210", KNOWN)


def test_known_misread_of_a_long_value():
    assert _known("priya.snama@example.com", KNOWN)        # two letters wrong
    assert _known("akiai0sfodnn7example", KNOWN)           # 0 for O
    assert not _known("rahul.verma@example.org", KNOWN)    # a different secret


def test_known_numbers_are_never_fuzzy():
    assert not _known("9876543211", KNOWN)      # one digit off is a different account
    assert not _known("98-765-43211", KNOWN)    # no letters at all


def test_known_short_values_are_never_fuzzy():
    assert not _known("abcde12", {"abcde1"})
    assert _known("abcde12", {"abcde12"})
