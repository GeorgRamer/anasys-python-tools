import xarray as xr
from collections import namedtuple
from dateutil.parser import parse as timeparser
import numpy as np


class LabelSorter:

    def __init__(self, preferences):
        self.sortorder = preferences.copy()
    
    def __call__(self, value):
        if value not in self.sortorder:
            self.sortorder.append(value)
        return self.sortorder.index(value)


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
                  which members of the `Tags` attributes need to match
                  
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
    ypix = np.arange(image.SampleBase64.shape[0])
    xpix = np.arange(image.SampleBase64.shape[1])
    transform = image.get_transform(global_coords=True, 
                                    mtransform=False)
    xmesh, ymesh = np.meshgrid(xpix, ypix)
    xpos, ypos = pix_to_xy(xmesh, ymesh, transform)
    xcoord = xr.DataArray(xpos, dims=("ypix","xpix"),
                      coords=(("ypix", ypix), ("xpix", xpix)))
    ycoord = xr.DataArray(ypos, dims=("ypix","xpix"),
                      coords=(("ypix", ypix), ("xpix", xpix)))
    
    arr = xr.DataArray(image.SampleBase64,
                       dims=("ypix","xpix"),
                       coords={"xpix":("xpix", xpix), "ypix":("ypix", ypix), 
                              "xpos": xcoord, "ypos": ycoord
                              })
    arr.attrs["TimeStamp"] = timeparser(image.TimeStamp)
    arr.attrs["transform"] = transform
    arr.attrs["Label"] = image.Label + " ({})".format(image.Tags["TraceRetrace"])
    if include_name:
        return image.DataChannel, arr
    return arr
    

def imagelist_to_Dataset(image_list):
    """convert list of images to a xarray.Dataset
    
    image_list: list of images
    
    returns xarray.Dataset with dims xpix and ypix and coordinates xy of the image position"""
    
    data_vars = [image_to_DataArray(img, True) for img in image_list]
    attrs = image_list[0].Tags
    attrs["TimeStamp"] = timeparser(image_list[0].TimeStamp)
    attrs["transform"] = image_list[0].get_transform(global_coords=True, 
                                                     mtransform=False)
    attrs["Label"] = image_list[0].Label+ " ({})".format(image_list[0].Tags["TraceRetrace"])
    return xr.Dataset(data_vars=dict(data_vars), attrs=attrs)


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
    chans["Background"] =  xr.DataArray(spectrum.Background.signal, dims=("wavenumbers"), coords=(("wavenumbers", spectrum.Background.wn),))
    chans.update(attrs_to_DataArray_dict(spectrum))
    ds =  xr.Dataset(chans)
    #ds = ds.assign_coords({"X":("x_pos", [ds["Location.X"].values]),"Y": ("y_pos",[ds["Location.Y"].values])})
    return ds

def spectra_list_to_Dataset(spectra_list):
    return xr.concat([spectrum_to_Dataset(spectrum) for spectrum in spectra_list], dim="spectral_index", coords="all", data_vars="all").drop_dims("dim_0")