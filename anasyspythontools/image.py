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
matplotlib.use("TkAgg") #Keeps tk from crashing on final dialog open
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from . import anasysfile
import base64, struct



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

    def _plot(self, global_coords=False, **kwargs):
        """Generates a pyplot image of image for saving or viewing"""
        if global_coords:
            width = float(self.Size.X)
            height = float(self.Size.Y)
            X0 = float(self.Position.X)
            Y0 = float(self.Position.Y)
            axes = [X0 - width/2, X0 + width/2, Y0 - height/2, Y0 + height/2]
        else:
            axes = [0, float(self.Size.X), 0, float(self.Size.Y)]
        #Set color bar range to [-y, +y] where y is abs(max(minval, maxval)) rounded up to the nearest 5
        imshow_args = { 'interpolation':'none', 'extent':axes}
        imshow_args.update(kwargs)
        # configure style if specified
        if "style" in imshow_args.keys():
            plt.style.use(imshow_args.pop("style"))
        img = plt.imshow(self.SampleBase64, **imshow_args)
        #Set titles
        plt.xlabel('μm')
        plt.ylabel('μm')
        #Adds color bar with units displayed
        units = self.Units
        if self.UnitPrefix != {}:
            units = self.UnitPrefix + self.Units
        #Set window title
        plt.gcf().canvas.set_window_title(self.Label)
        return plt

    def show(self, global_coords=False, **kwargs):
        """
        Opens an mpl gui window with image data. Options are documented:
        https://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.imshow
        Style can be specified with 'style' flag. Options:
        pyplot.style.options:
        https://tonysyu.github.io/raw_content/matplotlib-style-gallery/gallery.html
        """
        if type(self.SampleBase64) == dict:
        #Don't do anything if list is empty
            print("Error: No image data in Image object")
            return
        #Do all the plotting
        img = self._plot(global_coords=global_coords,**kwargs)
        #Display image
        img.show()

    def savefig(self, fname='', **kwargs):
        """
        Gets the plot from self._plot(), then saves. Options are documented:
        https://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.savefig
        """
        if type(self.SampleBase64) == dict:
        #Don't do anything if list is empty
            print("Error: No image data in Image object")
            return
        #Do all the plotting
        img = self._plot()
        #File types for save
        ftypes = (("Portable Network Graphics (*.png)", "*.png"),
                  ("Portable Document Format(*.pdf)", "*.pdf"),
                  ("Encapsulated Postscript (*.eps)", "*.eps"),
                  ("Postscript (*.ps)", "*.pdf"),
                  ("Raw RGBA Bitmap (*.raw;*.rgba)", "*.raw;*.rgba"),
                  ("Scalable Vector Graphics (*.svg;*.svgz)", "*.svg;*.svgz"),
                  ("All files", "*.*"))
        #Test for presense of filename and get one if needed
        if fname == '':
            fname = tk.filedialog.asksaveasfilename(filetypes=ftypes, defaultextension=".png", initialfile="Image.png")
        if fname == '':
            print("ERROR: User failed to provide filename. Abort save command.")
            return
        #If they made it this far, save (fname given)
        plt.savefig(fname, **kwargs)
 
