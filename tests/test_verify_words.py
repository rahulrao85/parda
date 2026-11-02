"""parda/verify.py COVER_WORDS: a verify finding that describes one of Parda's own covers is not a leak."""
import pytest

from parda.verify import COVER_WORDS


@pytest.mark.parametrize("finding", [
    "redacted value", "Redacted API key", "masked password", "row of asterisks", "****", "AWS key ********",
    "dots where the key was", "a row of dots", "●●●●●●", "hidden value", "private document", "covered name",
    "value under a cover",
])
def test_cover_descriptions_match(finding):
    assert COVER_WORDS.search(finding)


@pytest.mark.parametrize("secret", [
    "AKIAIOSFODNN7EXAMPLE", "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY", "sk-test-FAKE0000parda0000demo0000key",
    "priya.sharma@example.com", "ABCDE1234F", "postgres://admin:hunter2-not-real@db.example.com:5432/app",
    "TEST USER", "+91 98765 43210", "discover card 4111 1111 1111 1111", "recovery code 8812-3341",
    "dotson@example.com", "a*b*c",
])
def test_real_secrets_do_not_match(secret):
    assert not COVER_WORDS.search(secret)


def test_password_containing_a_cover_word_is_still_a_leak():
    assert not COVER_WORDS.search("hiddenvalley42")
