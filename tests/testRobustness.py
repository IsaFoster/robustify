from libraryTestCERN import testFunctions

def test_plus():
    assert testFunctions.plus([5,2,3]) == 10


def test_plus_decimals():
    assert testFunctions.plus([5.5,2.2,3.7]) == 11.4