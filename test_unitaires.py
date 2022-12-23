import pytest
import unittest


# def test_function():
#     assert f() == 4


# def test_latitude_degrees_range():
#     with pytest.raises(AssertionError):
#         position = script.Position(100, 100)
# # nous nous attendons à ce que le programme lève une erreur


def test_get_wordnet_pos(tag):
    assert get_wordnet_pos(CC) == "n"
#     assert get_wordnet_pos(DT) == wn.NOUN
#     assert get_wordnet_pos(EX) == wn.NOUN
#     assert get_wordnet_pos(FW) == wn.NOUN
#     assert get_wordnet_pos(IN) == wn.NOUN
#     assert get_wordnet_pos(JJ) == wn.ADJ
#     assert get_wordnet_pos(LS) == wn.NOUN
#     assert get_wordnet_pos(MD) == wn.NOUN
#     assert get_wordnet_pos(NNS) == wn.NOUN
#     assert get_wordnet_pos(PRP) == wn.NOUN
#     assert get_wordnet_pos(RBR) == wn.ADV
#     assert get_wordnet_pos(TO) == wn.NOUN
#     assert get_wordnet_pos(UH) == wn.NOUN
#     assert get_wordnet_pos(VBD) == wn.VERB
#     assert get_wordnet_pos(WRB) == wn.NOUN


# def test_cleaning(data):
#     assert cleaning(data)


# def test_my_predict(text):
#     assert my_predict(text)
