import pytest

from collect_and_annotate import license_ok


@pytest.mark.parametrize("name", ["CC BY 4.0", "CC BY-SA 3.0 de", "Public domain", "CC0", "PD"])
def test_accepts_open_licences(name):
    assert license_ok(name)


@pytest.mark.parametrize("name", ["CC BY-NC 2.0", "CC BY-ND 2.0", "CC BY-NC-SA 3.0",
                                  "CC BY-NC-ND 4.0", "Creative Commons NonCommercial", ""])
def test_rejects_nc_nd_and_empty(name):
    assert not license_ok(name)
