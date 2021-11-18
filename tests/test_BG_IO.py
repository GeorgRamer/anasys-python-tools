
import anasyspythontools as apt
import pytest
import glob
import os

TESTFOLDER = os.path.join(os.path.dirname(__file__),"test data")




@pytest.mark.parametrize("filename", glob.glob(os.path.join(TESTFOLDER, "*.irb"))) 
class TestBG:
    
    def setup_method(self, filename):
        pass
    
    def teardown_method(self):
        pass
    
    def test_basic_read(self, filename):
        f = apt.read(filename)
        assert f is not None
        assert isinstance(f, apt.irspectra.Background)
    
    def test_basic_read(self, filename):
        f = apt.read(filename)
    
    def test_check_equality(self, filename):
        assert apt.read(filename) == apt.read(filename)
    
    def test_check_attributes(self, filename):
        f = apt.read(filename)
        assert hasattr(f,"wn")
        assert hasattr(f,"signal")
