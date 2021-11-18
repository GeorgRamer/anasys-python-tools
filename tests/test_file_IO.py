import anasyspythontools as apt
import pytest
import glob
import os



class TestQuisIpsos:
    
    @pytest.mark.xfail
    def test_equality_fails(self):
        assert apt.read('test data/EmptyIRDoc.axd') == apt.read('BG_Full_stepped_test1.irb')



TESTFOLDER = "test data"
filenames =  glob.glob(os.path.join(TESTFOLDER, "*.axd"))\
                        +glob.glob(os.path.join(TESTFOLDER, "*.axz"))
print(filenames)
@pytest.mark.parametrize("filename", filenames)
class TestClass:
    
    def setup_method(self, filename):
        pass
    
    def teardown_method(self):
        pass
    
    def test_basic_read(self, filename):
        f = apt.read(filename)
        assert f is not None
        assert isinstance(f, apt.anasysdoc.AnasysDoc)
    
    def test_basic_read(self, filename):
        f = apt.read(filename)
    
    def test_check_equality(self, filename):
        assert apt.read(filename) == apt.read(filename)
 
    
