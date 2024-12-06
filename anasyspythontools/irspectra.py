# -*- encoding: utf-8 -*-
#
#  heightmap.py
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
import weakref
from .anasysfile import multi_element, simple_type
from . import repr_utils

import base64, tempfile

class IRRenderedSpectra(anasysfile.AnasysElement):
    """A data structure for holding Spectral data"""

    def __init__(self, irrenderedspectra, weakref_parent=None):
        self._weakref_parent = weakref_parent #parent object (Document)
        self._iterable_write = {}
        self._special_write = {'DataChannels': self._write_data_channels,
                               'FreqWindowMaps': self._write_freq_window_maps}
        self._special_read = {'DataChannels': self._get_data_channels,
                               'FreqWindowMaps': self._read_freq_window_maps,
                             "DataPoints":simple_type(int),
                              'StartWavenumber':simple_type(float),
                             "EndWavenumber":simple_type(float),
                             'OffsetWavenumber':simple_type(float), 
                             'Location': multi_element(float)}
        self._skip_on_write = ['Background'] #objects to skip when writing back to xml
        self._wrangle_data_channels(irrenderedspectra)
        self._wrangle_freqwindowmaps(irrenderedspectra)
        anasysfile.AnasysElement.__init__(self, etree=irrenderedspectra)
        self._Background = self._get_background() #get bg associated with this spectra

    def _wrangle_freqwindowmaps(self, irrenderedspectra):
        new_fwm = ET.SubElement(irrenderedspectra, 'FreqWindowMaps')
        for fwm in irrenderedspectra.findall('FreqWindowMap'):
            new_fwm.append(fwm)
            irrenderedspectra.remove(fwm)

    def _wrangle_data_channels(self, irrenderedspectra):
        new_datachannels = ET.SubElement(irrenderedspectra, 'temp_DataChannels')
        for dc in irrenderedspectra.findall('DataChannels'):
            dc.tag = 'DataChannel'
            new_datachannels.append(dc)
            irrenderedspectra.remove(dc)
        new_datachannels.tag = 'DataChannels'
    

    def _get_data_channels(self, datachannels):
        """Returns a dict of the DataChannel objects"""
        dcdict = {}
        for dc in datachannels:
            new_dc = DataChannel(dc)
            key = new_dc.Name
            key = self._check_key(key, dcdict)
            dcdict[key] = new_dc
        return dcdict

    def _read_freq_window_maps(self, freqwindowmaps):
        """Returns a list of FreqWindowMap's"""
        fwmlist = []
        for fwm in freqwindowmaps:
            new_fwm = anasysfile.AnasysElement(etree=fwm)
            fwmlist.append(new_fwm)
        return fwmlist

    def _write_freq_window_maps(self, elem, nom, freqwindowmaps):
        for fwm in freqwindowmaps:
            new_elem = fwm._anasys_to_etree(fwm, name='FreqWindowMap')
            elem.append(new_elem)

    def _write_data_channels(self, elem, nom, datachannels):
        for dc in datachannels.values():
            new_elem = dc._anasys_to_etree(dc, name="DataChannels")
            elem.append(new_elem)
    
    @property
    def Background(self):
        # this is a bit of risky solution:
        #   background can't be loaded during parsing, because it might not be parsed yet
        #   This stores it when it first requested. It only works, if the parent has not been GCed yet.
        if self._Background is None:
            self._Background = self._get_background()
        return self._Background
    
    def _get_background(self):
        try:
            return self._weakref_parent().Backgrounds[self.BackgroundID]
        except AttributeError:
            # _parent was not set or has been GCed
            pass
        return None
        
        
        
    def _repr_png_(self):
        with plt.ioff():
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            fig = matplotlib.figure.Figure(figsize=(3,2))
            canv = FigureCanvasAgg(fig)
            ax = canv.figure.add_subplot(111)
            ax.set_xlabel("wavenumbers / $\\mathrm{cm^{-1}}$")
            ax.invert_xaxis()
            idx = 0
            lins = []
            for channel_name, channel in self.DataChannels.items():
                if idx!=0:
                    ax = ax.twinx()
                idx+=1
                lins+=ax.plot(channel.wn, channel.signal, color=f"C{idx}", label=channel_name)
                ax.set_yticks([])
            fig.legend(lins, [l.get_label() for l in lins],handlelength=0, handletextpad=0, labelcolor="linecolor")

            canv.draw()
        with tempfile.TemporaryFile() as f:
            fig.savefig(f, format="png", bbox_inches="tight")
            f.seek(0)
            byts = f.read()
        return byts

    def _repr_html_content_(self):
        b64 =  base64.b64encode(self._repr_png_()).decode("utf8")
        return '<div style="float: left;"><img style="height:150px;" src="data:image/png;base64,{b64}"></div><div style="height:200px; overflow-y:hidden; overflow-y:auto; white-space:nowrap;"> {table}</div>'.format(b64=b64, 
        table=repr_utils.repr_tag_dict_html(self.attrs))
    
    def _repr_html_(self):
        return "<div>{content}</div>".format(
            content=self._repr_html_content_())

class DataChannel(anasysfile.AnasysElement):
    """Data structure for holding spectral Data"""

    def __init__(self, datachannels):
        anasysfile.AnasysElement.__init__(self, etree=datachannels)
    @property
    def wn(self):
        return float(self["Start"])+np.arange(len(self["SampleBase64"]))/float(self["Resolution"])
		
    @property
    def signal(self):
        return self["SampleBase64"]
        
        
        

        

class Background(anasysfile.AnasysElement):
    """Data structure for holding background data"""

    def __init__(self, background):
        self._special_write = {'Table': self._nparray_to_serial_tags,
                              'AttenuatorPower': self._nparray_to_serial_tags}
        self._special_read = {'Table': self._serial_tags_to_nparray,
                              'AttenuatorPower': self._serial_tags_to_nparray}
        anasysfile.AnasysElement.__init__(self, etree=background)
    
    @property
    def wn(self):
        vstart = float(self.StartWavenumber)
        vlen = len(self.Table)
        vres = self.IRSweepResolution
        if not isinstance(vres, str) and vres['{http://www.w3.org/2001/XMLSchema-instance}nil']:
            vend = float(self.EndWavenumber)
            return np.linspace(vstart, vend, vlen)
        else:
            vres = float(vres)
            return vstart + np.arange(vlen) * vres
        # return float(self.StartWavenumber) + np.arange(len(self.Table)) * float(self.IRSweepResolution)
    
    @property
    def signal(self):
        #Remark: for some reason (maybe to ensure loading/saving doesn't degrade the file?), 
        # originally, the power meter data was parsed as decimal. For ease of use, this function
        # return floats.
        try:
            return self.Table.astype(np.float64) / self.AttenuatorPower.astype(np.float64)
        except ValueError:
            # something went wrong while recording. return np.nan to ensure no crashes
            return np.nan
        
        
        

        
    #
    # def _write_table(self, elem, nom, table):
    #     print(self, elem, nom, table)
    #     elem.append()
    # def _get_table(self, table): # 126
    #     table_data = []
    #     for double in table:
    #         table_data.append(float(double.text))
    #         table.remove(double)
    #     table_data = np.array(table_data)
    #     return table_data

    # def _get_attenuatorPower(self, atpow):
    #     pewerdata = []
    #     for double in atpow:
    #         at
