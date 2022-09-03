from . import anasysfile
from . import anasysdoc
from . import heightmap
from . import image
from . import irspectra
from . import anasysio


__version__ = "0.4.0"

def read(fn):
    fr = anasysio.AnasysFileReader(fn)
    if fr._filetype == "full":
    	return anasysdoc.AnasysDoc(fr._doc)
    if fr._filetype == "bg":
    	return irspectra.Background(fr._doc)
