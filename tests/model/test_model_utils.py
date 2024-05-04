from model.utils import get_hash_of_two_strings


def test_get_hash_of_two_strings():
    string1 = "hello"
    string2 = "world"

    result = get_hash_of_two_strings(string1, string2)

    assert result == "k2oYXKqiZrucvpgengXLeM1zKwsygOuURBK7b4+PB68="
