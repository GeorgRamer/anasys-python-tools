from . import anasysfile
from . import anasysdoc
from . import heightmap
from . import image
from . import irspectra
from . import anasysio


__version__ = "0.5.1"

def read(fn):
    try: 
        fr = anasysio.AnasysFileReader(fn)
    except FileNotFoundError as e:
        raise FileNotFoundError("File not found: " + fn) from e
    if fr._filetype == "full":
        return anasysdoc.AnasysDoc(fr._doc)
    if fr._filetype == "bg":
        return irspectra.Background(fr._doc)
