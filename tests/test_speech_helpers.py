"""parda/speech.py pure helpers: the email pattern, the deterministic backstop rules and alignment words."""
import pytest

from parda.speech import EMAIL, _roman, _rules, _words


@pytest.mark.parametrize("said, want", [
    ("priya.sharma@example.com", "priya.sharma@example.com"),
    ("or email priya dot sharma at example dot com.", "priya dot sharma at example dot com"),
    ("priya.charma at example.com", "priya.charma at example.com"),
    ("rahul at mail dot co dot in", "rahul at mail dot co dot in"),
    ("write to test_user@sub.example.org now", "test_user@sub.example.org"),
])
def test_email_matches_written_and_spoken(said, want):
    m = EMAIL.search(said)
    assert m and m.group() == want


@pytest.mark.parametrize("said", ["meet me at noon", "look at example", "it costs 5 dot 50", "version 1.2.3",
                                  "see the dotcom era"])
def test_email_ignores_ordinary_speech(said):
    assert not EMAIL.search(said)


def test_words_spell_digits_and_drop_punctuation():
    assert _words("Call me on 98765, it's done.") == ["call", "me", "on", "nine", "eight", "seven", "six", "five",
                                                       "its", "done"]


def test_words_devanagari_numbers_and_romanisation():
    assert _words("नाइन एट सेवन") == ["nine", "eight", "seven"]
    assert _words("एक दो तीन") == ["ek", "do", "teen"]
    assert _words("मेरा नंबर") == ["mera", _roman("नंबर")]
    assert _words("ज़ीरो") == ["zero"]  # the nukta is ignored when looking up a number word
    assert _words("") == []


@pytest.mark.parametrize("word, want", [("मेरा", "mera"), ("कमल", "kamal"), ("अमित", "amit"), ("ज़रा", "jara"),
                                        ("क्या", "kya")])
def test_roman(word, want):
    assert _roman(word) == want


@pytest.mark.xfail(reason="anusvara before a labial romanises as 'n': _roman('नंबर') gives 'nanbar', its docstring "
                          "promises 'nambar'", strict=True)
def test_roman_docstring_example():
    assert _roman("नंबर") == "nambar"


def test_rules_find_email_and_digit_run():
    turns = [{"text": "Okay, quick note. Call me on nine eight seven six five four three two one zero "
                      "or email priya.charma at example.com"}]
    out = _rules(turns)
    assert (0, ["priya", "charma", "at", "example", "com"], "email", "email pattern (rule)") in out
    digits = [r for r in out if r[2] == "number"]
    assert digits == [(0, "nine eight seven six five four three two one zero".split(), "number",
                       "10 spoken digits (rule)")]


def test_rules_digit_threshold_and_turn_index():
    turns = [{"text": "one two three four"}, {"text": "my PIN is 4 4 1 2 9 thanks"},
             {"text": "पिन है एक दो तीन चार पांच"}]
    out = _rules(turns)
    assert [(i, kind, len(w)) for i, w, kind, _ in out] == [(1, "number", 5), (2, "number", 5)]


def test_rules_nothing_private():
    assert _rules([{"text": "Nothing to see here, just a demo of the settings page."}]) == []
    assert _rules([]) == []
