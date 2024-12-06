import xarray as xr
from collections import namedtuple

import numpy as np
from collections import namedtuple
import numbers
import re
import math
from . import xarray_utils




class LabelSorter:

    def __init__(self, preferences):
        self.sortorder = preferences.copy()
    
    def __call__(self, value):
        if value not in self.sortorder:
            self.sortorder.append(value)
        return self.sortorder.index(value)



def split_unit_float(value_str):
    """Does nothing if value is numeric. 
       Otherwise attempts to convert value float"""
    if isinstance(value_str, numbers.Number):
        return value_str
    if value_str is None or value_str=="":
        return float("nan")
    num_part = re.split(r"[\s%]+", value_str.strip(" "))[0]
    return float(num_part)
    

def timeparser(timestr):
    """parses a time string into a numpy.datetime64 object. 
    This strips the time zone information (because it is easier
    and timezones are unlikely to matter for AFM images.
    """
    
    # This code forces nanosecond resolution (although not)
    # provided by the AFM, because that is required by xarray
    return np.datetime64(timestr.split("+")[0], 'ns')

def first_float(arr):
    """
    Returns first element of array and makes sure it's a float
    """
    return float(arr[0])



class ExportSettingsReg:

    def __init__(self, coords, attrs):
        self._attrs= attrs.copy()
        self._coords = coords.copy()
        
    def add_coord(self, name, type_conversion):
        self._coords[name] = type_conversion
    
    def del_coord(self, name):
        if name in self._coords:
            del self._coords[name]
    
    def add_attr(self, name, type_conversion):
        self._attrs[name] = type_conversion
    
    def del_coord(self, name):
        if name in self._attrs:
            del self._attrs[name]
    @property
    def coords(self):
        return self._coords.copy()
    @property
    def attrs(self):
        return self._attrs.copy()
        
        
        
    def create_coord_dict(self, anasys_element):
        attrs = anasys_element.attrs
        return {k: v(attrs.get(k,None)) for k,v in self.coords.items() if k in self.coords}
    
    def create_attr_dict(self, anasys_element):
        attrs = anasys_element.attrs
        return {k:  v(attrs.get(k,None)) for k,v in self.attrs.items() if k in self.attrs}        


_init_height_maps_coords = {
    "TimeStamp":                    timeparser,
    "Tags.ScanRate":                split_unit_float, 
    "Tags.Setpoint":                split_unit_float,
    "Tags.IGain":                   float,
    "Tags.PGain":                   float,
    "Tags.ScanMode":                str,
    "Tags.TraceRetrace":            str,
    "Tags.ACDriveEnabled":          bool,
    "Tags.ACDriveFrequency":        split_unit_float,
    "Tags.ACDriveAmplitude":        split_unit_float,
    "Tags.IRWavenumber":            split_unit_float,
    "Tags.IRAttenuation":           split_unit_float,
    'Tags.IRPowerMaximizerEnabled': bool,
    'Tags.IRPolarization':          split_unit_float,
    'Label':                        str,
}

_init_height_maps_attrs = {}#"Tags.ScanMode":str, "Tags.TraceRetrace":str}


ExportSettingsHeightMap = ExportSettingsReg(coords = _init_height_maps_coords,
                                            attrs = _init_height_maps_attrs)



_init_rendered_spectra_coords = {
    'Label':              str,
    'Location.X':         float,
    'Location.Y':         float,
    'PulseRate':          float,
    'TimeStamp':          timeparser,
    'AFMMode':            str,
    'AttenuationBase64':  first_float,
    'BackgroundFileName': str,
    'BackgroundOption':   str,
    'Polarization':       str,
    # 'Background': None,
    # 'BackgroundID': None,
    # 'BandPassEnabled': None,
    # 'BeamShapeFactorBase64': None,
    # 'ChannelZero': None,
    # 'CoAverages': None,
    # 'DataChannels': None,
    # 'DataPoints': None,
    # 'DetectorGain': None,
    # 'DigitalGain': None,
    # 'DutyCycle': None,
    # 'EndWavenumber': None,
    # 'FreqRMSZero': None,
    # 'FreqWindowData': None,
    # 'FreqWindowMaps': None,
    # 'FrequencySearchLocation': None,
    # 'FrequencySearchWidth': None,
    # 'Lockin2AmpZero': None,
    # 'Lockin2RTAmpZero': None,
    # 'LockinAmpZero': None,
    # 'LockinRTAmpZero': None,
    # 'OAPFocusPos': None,
    # 'OffsetWavenumber': None,
    # 'PeakIZero': None,
    # 'PeakZero': None,
    # 'PowerMaximizerEnabled': None,
    # 'ProbePower': None,
    # 'PulseRate': None,
    # 'RMSZero': None,
    # 'RotaryPolarizerMotorPositionBase64': None,
    # 'SampleRate': None,
    # 'SampleSize': None,
    # 'StageZPos': None,
}


_init_rendered_spectra_attrs = {}


ExportSettingsRenderedSpectra = ExportSettingsReg(coords = _init_rendered_spectra_coords,
                                            attrs = _init_rendered_spectra_attrs)                                            



class AttrsCDFify:
    
    def __init__(self, attrs_conversion):
        self._attrs_conversion = attrs_conversion.copy()
    
    def _convert_attr(self, name, value, to_cdf=True):
        sel_converter = 0
        if not to_cdf:
            sel_converter = 1
        if name in self._attrs_conversion:
            return (name, self._attrs_conversion[name][sel_converter](value))
        return None
    
    def _convert_array(self, array, to_cdf=True):
        converted = [self._convert_attr(name, value, to_cdf) for name,value in array.attrs.items()]
        return array.assign_attrs( dict([c for c in converted if c is not None]))
    
    
    def _convert(self, xarray_obj, to_cdf=True):
        xarray_obj = self._convert_array(xarray_obj, to_cdf)
        if isinstance(xarray_obj, xr.Dataset):
            old_attr = xarray_obj.attrs
            xarray_obj = xarray_obj\
                            .map(self._convert_array, args=[to_cdf], keep_attrs=False)\
                            .assign_attrs(old_attr)
            
        return xarray_obj
    
    def cdfify(self, xarray_obj):
        """
        returns an xarray_object that can be stored as netcdf using h5netcdf.
        """
        return self._convert(xarray_obj, to_cdf=True)
    
    def uncdfify(self, xarray_obj):
        """
        undoes changes done by cdfify.
        """
        return self._convert(xarray_obj, to_cdf=False)
    
    def set_attr(self, name, to_cdf_fun, from_cdf_fun):
        self._attrs_conversion[name] = [to_cdf_fun, from_cdf_fun]
        
    def del_attr(self, name):
        del self._attrs_conversion[name]
        

_init_attrscdfify = {"TimeStamp":( str, np.datetime64)}

attrsCDFify = AttrsCDFify(_init_attrscdfify)

cdfify = attrsCDFify.cdfify
uncdfify = attrsCDFify.uncdfify



def get_concurrent_images(img_list, matched_attrs=["TimeStamp"], 
                          matched_tags=["TraceRetrace"]):
    
    """
    Find all images that belong together, typically because they have the same TimeStamp
    
    Parameters:
    ----------
    
    img_list: list of anasys images
    matched_attrs: list of str, default ["TimeStamp"]
                    which attributes need to match for images to be considered "concurrent"
    matched_tags: list of str, default ["TraceRetrace"]
                  which elements of the `Tags` attributes need to match
                  
    returns:
    --------
    
    concurrent_img_dict: dictionary of concurrent images. 
                        Keys are named tuples of matched_attrs and matched_tags
                        Values are lists of images
    """
    nt = namedtuple("map_properties", ", ".join(matched_attrs+matched_tags))
    label_sorter = LabelSorter(["height"])
    
    concurrent_img_dict = {}
    for img in img_list:
        img_id= nt(**{attr:getattr(img, attr) for attr in matched_attrs},
                   **{tag:img.Tags[tag] for tag in matched_tags})
        if img_id in concurrent_img_dict:
            concurrent_img_dict[img_id].append(img)
        else:
            concurrent_img_dict[img_id] = [img]
    for img_list in concurrent_img_dict.values():
        img_list.sort(key=lambda img: label_sorter(img.DataChannel))
    return concurrent_img_dict


def pix_to_xy(xpix, ypix, transform_matrix):
    pix = np.vstack([xpix.flatten(), ypix.flatten(), np.ones(xpix.shape).flatten()])
    matr = transform_matrix@pix
    return matr[0].reshape(xpix.shape), matr[1].reshape(xpix.shape)






    

def image_to_DataArray(image, include_name=False):
    """ export a HeightMap or Image to a DataArray"""
    
    
    ypix = np.arange(image.SampleBase64.shape[0])
    xpix = np.arange(image.SampleBase64.shape[1])
    
    is_RGB = len( image.SampleBase64.shape)==3
    
    dims = ["y", "x"]
    coords = {"xpix":("x", xpix), "ypix":("y", ypix)}
    
    if is_RGB:
        dims.append("color")
        coords.update({"color":["R","G","B", "A"]})
    
    transform = image.get_transform(global_coords=True, 
                                    mtransform=False)
    
   
    
    arr = xr.DataArray(image.SampleBase64,
                       dims=dims,
                       coords=coords)
                              
    
    arr.attrs["TimeStamp"] = timeparser(image.TimeStamp)
    arr.attrs["transform"] = transform
    label =  image.Label
    if not is_RGB:
        label += " ({})".format(image.Tags["TraceRetrace"])
    arr.attrs["Label"] = label
    
    arr = arr.assign_coords(ExportSettingsHeightMap.create_coord_dict(image))
    arr = arr.assign_attrs(ExportSettingsHeightMap.create_attr_dict(image))
    
    
    if include_name:
        return image.DataChannel, arr
    return arr
    

def imagelist_to_Dataset(image_list):
    """convert list of images to a xarray.Dataset
    
    image_list: list of images, this 
    
    returns xarray.Dataset with dims xpix and ypix and coordinates xy of the image position"""
    data_vars = [image_to_DataArray(img, True) for img in image_list]
    return xr.Dataset(data_vars=dict(data_vars), coords=data_vars[0][1].coords.copy(), attrs=data_vars[0][1].attrs.copy())


def attr_to_DataArray(spectrum):
    for attr, val in spectrum.attrs.items():
        if attr in ["DataChannels", "Background"]:
            continue 
        if not isinstance(val, (dict,np.ndarray)):
            yield attr,  xr.DataArray(np.array(val))
        elif isinstance(val, dict):
            for k, v in val.items():
                yield  "{}.{}".format(attr,k), v


def attrs_to_DataArray_dict(spectrum):
    return dict(attr_to_DataArray(spectrum))
            
        

def channel_to_DataArray(channel):
    arr = xr.DataArray(channel.signal, dims=("wavenumbers"), coords=(("wavenumbers", channel.wn),))
    return arr


def spectrum_to_Dataset(spectrum):
    chans = {channel:channel_to_DataArray(spectrum.DataChannels[channel]) for channel in spectrum.DataChannels}
    attenuation = spectrum["AttenuationBase64"]
    for channame, chan in chans.items():
        if chan.wavenumbers.data.shape == attenuation.shape:
            wn = list(chans.values())[0].wavenumbers.data
            chans["Attenuation"] =  xr.DataArray(spectrum["AttenuationBase64"], dims=("wavenumbers"), coords=(("wavenumbers", wn),))
            break
    else:
        # as a fall back: try to make it fit to background
        if attenuation.shape == spectrum.Background.wn.shape:
            chans["Attenuation"] =  xr.DataArray(attenuation, dims=("wavenumbers"), coords=(("wavenumbers", spectrum.Background.wn),))
        else:
            # and finally, just add nans instead
            chans["Attenuation"] =  xr.DataArray(spectrum.Background.wn*np.nans, dims=("wavenumbers"), coords=(("wavenumbers", spectrum.Background.wn),))
    chans["Background"] =  xr.DataArray(spectrum.Background.signal, dims=("wavenumbers"), coords=(("wavenumbers", spectrum.Background.wn),))
    
    ds =  xr.Dataset(chans).assign_attrs(ExportSettingsRenderedSpectra.create_attr_dict(spectrum))\
            .assign_coords(ExportSettingsRenderedSpectra.create_coord_dict(spectrum))

    return ds


def spectra_list_to_Dataset(spectra_list):
    return xr.concat([spectrum_to_Dataset(spectrum) for spectrum in spectra_list], dim="spectral_index", coords="all", data_vars="all")
