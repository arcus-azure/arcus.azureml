import setuptools 

def test_package_discovery():
    assert len(setuptools.find_packages('arcus')) > 0
