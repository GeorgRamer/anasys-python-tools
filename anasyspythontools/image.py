# -*- encoding: utf-8 -*-
#
#  image.py
#
#  Copyright 2017 Cody Schindler <cschindler@anasysinstruments.com>
#
#  This program is the property of Anasys Instruments, and may not be
#  redistributed or modified without explict permission of the author.

import xml.etree.ElementTree as ET
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from . import anasysfile
import base64, struct, tempfile
from .plotting import imshow_transformable, RobustNormalize
from matplotlib.transforms import Affine2D

class Image(anasysfile.AnasysElement):
    """A data structure for holding Image data"""

    def __init__(self, image):
        # self._parent = parent
        self._iterable_write = {}
        self._special_write = {'Tags': self._write_tags}
        self._skip_on_write = []
        self._special_read = {'Tags': self._read_tags}
        anasysfile.AnasysElement.__init__(self, etree=image)
        #Rearrange data into correct array size
        self.SampleBase64 = self.SampleBase64.reshape(int(self.Resolution.Y), int(self.Resolution.X),-1)
    
    def _decode_bs64(self, data):
        """Returns base64 data decoded in a numpy array"""
        if data is None:
            return np.ndarray(0)
        decoded_bytes = base64.b64decode(data.encode())
        fmt = 'B'*int((len(decoded_bytes)))
        structured_data = struct.unpack(fmt, decoded_bytes)
        decoded_array = np.array(structured_data)
        return decoded_array
    
    def _encode_bs64(self, np_array):
        """Returns numpy array encoded as base64 string"""
        raise NotImplementedError
    
    def _write_tags(self, elem, nom, tags):
        new_elem = ET.SubElement(elem, nom)
        for k, v in tags.items():
            tag = ET.SubElement(new_elem, "Tag")
            tag.set("Name", k)
            tag.set("Value", v)

    def _read_tags(self, element):
        """Turn tags into a dict of dicts"""
        tag_dict = {}
        for tag in list(element):
            tag_dict[tag.get('Name')] = tag.get('Value')
        return tag_dict

    # def _tags_to_etree(self, tags_obj):
    #     """Converts tags back to xml"""
    #     root = ET.Element("Tags")
    #     for k, v in tags_obj:
    #         sub = ET.SubElement(root, "Tag")
    #         sub.set("Name", k)
    #         sub.set("Value", v)
    #     return root

    def show(self, global_coords=False,ax=None,axes_style="conventional",title=None, **kwargs):
        """Generates a pyplot image of image for saving or viewing"""
        
        #Set color bar range to [-y, +y] where y is abs(max(minval, maxval)) rounded up to the nearest 5
        imshow_args = { 'interpolation':'none', 'datatransform':self.get_transform(global_coords=global_coords)}
        imshow_args.update(kwargs)
        if ax is None:
            _, ax = plt.subplots()
            
        imshow_args["ax"] = ax
        img = imshow_transformable(self.SampleBase64, **imshow_args)
        self.apply_axes_styling(ax, style=axes_style)
        #Set titles
        if title is None:
            ax.set_title(self.Label)
        else:
            ax.set_title(title)
        return img
        
    def apply_axes_styling(self, ax, style=None):
        if style is None:
            return
        elif style == "microscopy":
            from matplotlib_scalebar.scalebar import ScaleBar
            # remove axis labels, add scale bar, remove axis frame
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_xticks([])
            ax.set_yticks([])
            sbar = ScaleBar(1, "µm", location="lower right")
            ax.add_artist(sbar)
        elif style == "conventional":
            # add axes labels and ticks
            #Set titles
            ax.set_xlabel('μm')
            ax.set_ylabel('μm')
        else:
           raise ValueError(f"Unknown styling {style}")
   
    def get_transform(self, global_coords=False, mtransform=True):
        """returns affine transform between pixels real world dimensions
        global_coords : default False, also apply shift and rotation
        mtransform : default True
            True : return as matplotlib.transforms.Affine2D
            False: return as 3x3 numpy array"""
        scale_x = float(self.Size.X)/float(self.Resolution.X)
        # scal_y is negative because the pixels are top to bottom
        scale_y = -float(self.Size.Y)/float(self.Resolution.Y)
        # no docs for anasys rotation. need to check if this is correct
        rotation = 0.0
        translation_x = 0
        translation_y = 0
        if global_coords:
            translation_x = float(self.Position.X) -  float(self.Size.X)/2
            translation_y = float(self.Position.Y) +  float(self.Size.Y)/2
        M = np.zeros((3,3))
        M[2,2] = 1
        M[0,0] = 1
        M[1,1] = 1
        M_scale = M.copy()
        M_scale[0,0] = scale_x
        M_scale[1,1] = scale_y
        M_rotation = M.copy()
        M_rotation[0,:2] = [np.cos(rotation), -np.sin(rotation)]
        M_rotation[1,:2] = [np.sin(rotation), np.cos(rotation)]
        M_translation = M.copy()
        M_translation[0,2] = translation_x
        M_translation[1,2] = translation_y
        M_trans = M_translation@M_rotation@M_scale
        if mtransform:
            return Affine2D(M_trans)
        return M_trans

    def get_y(self, global_coords=False):
        height = float(self.Size.Y)
        Y0 = float(self.Position.Y)
        y_pixels = int(self.Resolution.Y)
        if global_coords:
            return np.linspace(Y0 - height/2, Y0 + height/2, y_pixels)
        else:
            return np.linspace(0, height, y_pixels)
            
    def get_x(self, global_coords=False):
        width = float(self.Size.X)
        X0 = float(self.Position.X)
        x_pixels = int(self.Resolution.X)
        if global_coords:
            return np.linspace(X0 - width/2, X0 + width/2,x_pixels)
        else:
            return np.linspace(0, width, x_pixels)        
        



    def _repr_png_(self):
        with plt.ioff():
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            fig = matplotlib.figure.Figure(figsize=(2,2))
            canv = FigureCanvasAgg(fig)
            ax = canv.figure.add_subplot(111)
            self.show(ax=ax, global_coords=False, axes_style="microscopy", title="")
            ax.set_xticks([])
            ax.set_yticks([])
            canv.draw()
        with tempfile.TemporaryFile() as f:
            fig.savefig(f, format="png", bbox_inches="tight")
            f.seek(0)
            byts = f.read()
        return byts

    def _repr_html_content_(self):
        b64 =  base64.b64encode(self._repr_png_()).decode("utf8")
        return "<img src='data:image/png;base64,{b64}'></div>".format(b64=b64)
    
    def _repr_html_(self):
        return "<div>{content}</div>".format(
        content=self._repr_html_content_())

 
