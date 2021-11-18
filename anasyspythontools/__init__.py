from . import anasysfile
from . import anasysdoc
from . import heightmap
from . import image
from . import irspectra
from . import anasysio

def read(fn):
    doc = anasysio.AnasysFileReader(fn)._doc
    if doc._filetype == "full":
    	return anasysdoc.AnasysDoc(doc)
    if doc._filetype == "bg":
    	return irspectra.Background(doc._etree)
