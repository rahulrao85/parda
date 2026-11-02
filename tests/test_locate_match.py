"""parda/locate.py: value_of strips a label Spark included, _match finds the value among a line's characters."""
import pytest

from parda.locate import _match, norm, value_of


@pytest.mark.parametrize("text, want", [
    ("AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE", "AKIAIOSFODNN7EXAMPLE"),
    ("AWS_SECRET_ACCESS_KEY = wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY", "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"),
    ("Name: TEST USER", "TEST USER"),
    ("Date of birth: 01-Jan-1990", "01-Jan-1990"),
    ("AKIAIOSFODNN7EXAMPLE", "AKIAIOSFODNN7EXAMPLE"),
    ("PIN: 12", "PIN: 12"),               # value too short to be sure it is not part of the text
    ("priya.sharma@example.com", "priya.sharma@example.com"),
    ("", ""),
])
def test_value_of(text, want):
    assert value_of(text) == want


def line(text):
    return {"text": text, "chars": [], "box": [0, 0, 10 * len(text), 20]}


def test_match_exact_after_the_label():
    ln = line("AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE")
    i, j = _match(ln, "AKIAIOSFODNN7EXAMPLE")
    assert norm(ln["text"])[i:j] == "akiaiosfodnn7example"
    assert _match(ln, "AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE") == (i, j)  # label in Spark's text is dropped


def test_match_uses_ocr_chars_when_present():
    # the recogniser's characters win over the line text (here the text lost its leading label)
    ln = {"text": "ABCDE1234F", "chars": [[c, 10.0 * k] for k, c in enumerate("ID ABCDE1234F")], "box": [0, 0, 130, 20]}
    assert _match(ln, "ABCDE1234F") == (2, 12)


def test_match_value_glued_to_label_by_ocr():
    # OCR glued the label to the value (PANABCDE1234F): with Spark's kind known, the whole token is the value
    assert _match(line("PANABCDE1234F"), "ABCDE1234F", kind="id_number") == (0, 13)
    assert _match(line("PANABCDE1234F"), "ABCDE1234F") is None  # without a kind, a piece could be a different value


def test_match_respects_token_boundaries():
    assert _match(line("Balance -25,000.00"), "-5,000.00") is None
    assert _match(line("Debit -5,000.00 today"), "-5,000.00") is not None
    assert _match(line("x12345678y"), "12345678") == (1, 9)  # a letter glued to a number is a label, not more number
    i, j = _match(line("ABCDE1234F and BCDE1"), "BCDE1")  # skips the embedded hit, finds the token
    assert norm("ABCDE1234F and BCDE1")[i:j] == "bcde1" and i > 5


def test_match_fuzzy_for_long_values_only():
    i, j = _match(line("AWS_ACCESS_KEY_ID=AKIAI0SFODNN7EXAMPLE"), "AKIAIOSFODNN7EXAMPLE")  # OCR 0 for O
    assert j - i == 20
    assert _match(line("code AB0DE"), "ABCDE") is None                          # short: exact only
    assert _match(line("AWS_ACCESS_KEY_ID=ZZZZZZZZZZZZZZZZZZZZ"), "AKIAIOSFODNN7EXAMPLE") is None


def test_match_value_being_typed():
    ln = line("Email: priy")
    i, j = _match(ln, "priya.sharma@example.com")
    assert norm(ln["text"])[i:j] == "priy"
    assert _match(line("Email: pri"), "priya.sharma@example.com") is None  # 3 characters say nothing yet


def test_match_short_values_only_as_whole_line():
    assert _match(line("25"), "25") == (0, 2)
    assert _match(line("Day 25"), "25") is None
    assert _match(line("x"), "x") is None


def test_match_password_tolerates_misreads():
    ln = line("Password vQAVbxx^P6k6")
    assert _match(ln, "QAVbxx*P6Kk", kind="password") is not None
    assert _match(ln, "QAVbxx*P6Kk", kind="api_key") is None
    assert _match(line("Mobile numbe"), "Mobile7", kind="password") is None
