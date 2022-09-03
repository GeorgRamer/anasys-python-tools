import xarray as xr
from collections import namedtuple
from dateutil.parser import parse as timeparser
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
    num_part = re.split("[.\s%]+", value_str.strip(" "))[0]
    return float(num_part)
    





class ExportSettingsReg:

    def __init__(self, coords, attrs):
        self._attrs= attrs.copy()
        self._coords = coords.copy()
        
    def add__coord(self, name, type_conversion):
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
        return {k: v(attrs[k]) for k,v in self.coords.items() if k in self.coords}
    
    def create_attr_dict(self, anasys_element):
        attrs = anasys_element.attrs
        return {k:  v(attrs[k]) for k,v in self.attrs.items() if k in self.attrs}        



_init_height_maps_coords = {"Tags.ScanRate":split_unit_float, "Tags.Setpoint":split_unit_float,
        "Tags.IGain":float, "Tags.PGain":float, "Tags.ACDriveEnabled":bool, "Tags.ACDriveFrequency":split_unit_float, 
        "Tags.ACDriveAmplitude":split_unit_float, "TimeStamp":timeparser}

_init_height_maps_attrs = {"Tags.ScanMode":str, "Tags.TraceRetrace":str}


ExportSettingsHeightMap = ExportSettingsReg(coords = _init_height_maps_coords,
                                            attrs = _init_height_maps_attrs)



_init_rendered_spectra_coords = {'Location.X':float,
                        'Location.Y':float,
                        'PulseRate':float,
                        'TimeStamp':timeparser}
_init_rendered_spectra_attrs = {'Label':str}


ExportSettingsRenderedSpectra = ExportSettingsReg(coords = _init_rendered_spectra_coords,
                                            attrs = _init_rendered_spectra_attrs)                                            




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






class AffineProjectionHandler:


    def __init__(self, transform):
        self.dimensionality = 2
        self.transform = transform
        self._transform_changed()
    
    def _transform_changed(self):
        self.decomposed_transform = self.decompose_transform(self.transform)
        self.is_seperable = self.has_seperable_dimensions(self.decomposed_transform)
        
    def decompose_transform(self, transform):
        scale =  np.sqrt(np.sum(transform ** 2, axis=0))[:self.dimensionality]
        translation = transform[0:self.dimensionality, self.dimensionality]
        rotation = math.atan2(transform[1, 0], transform[0, 0])
        shear = math.atan2(- transform[0, 1], transform[1, 1]) - rotation
        shear = np.mod(shear, np.pi)
        return namedtuple("affine_components", "scale translation rotation shear")(scale=scale,
                 translation = translation, rotation = rotation, shear = shear)
    
    def compose_transform(self, decomposed_transform, ignore_shear_rotate=False):
        
        sx, sy =  decomposed_transform.scale
        translation = decomposed_transform.translation
        rotation = decomposed_transform.rotation
        shear = decomposed_transform.shear
        if ignore_shear_rotate:
            rotation = 0
            shear = 0
        trans = np.array([
                [sx * np.cos(rotation), -sy * np.sin(rotation + shear), 0],
                [sx * np.sin(rotation),  sy * np.cos(rotation + shear), 0],
                [                      0,                                0, 1]
            ])
        trans[0:2, 2] = translation
        return trans
        
    def get_dimmed_transform(self, in_dim, in_vars, out_dim, out_vars, ignore_shear_rotate=False):
        trans = self.compose_transform(self.decomposed_transform,
             ignore_shear_rotate=ignore_shear_rotate)
        return xr.DataArray(trans, dims=(in_dim, out_dim), coords={in_dim:list(in_vars)+["const"], out_dim:list(out_vars)+["const"]})
       
    def has_seperable_dimensions(self, decomposed_transform):
         if decomposed_transform.shear != 0:
             return False
         if not (np.isclose(np.mod(decomposed_transform.rotation, np.pi/4),0)):
             return False
         return True
    
    def _perform_projection(self, coord_array, ignore_shear_rotate):
        "performs projection. assumes 'spatial_var' is the dim containing spatial variables"
        coord_array=coord_array.transpose(..., "spatial_var")
        
        
        trans =  self.get_dimmed_transform(in_dim="spatial_var", in_vars=coord_array.coords["spatial_var"].to_numpy(), 
                            out_dim="spatial_var_out", out_vars=coord_array.coords["spatial_var"].to_numpy(),
                            ignore_shear_rotate=ignore_shear_rotate)
        
        return (trans @ coord_array).rename({"spatial_var_out":"spatial_var"})
    
    #def _project_1ds_1ds(self, coord1, coord2, coord1_outname=None, coord2_outname=None, ignore_shear_rotate=False):
    #    if not self.is_seperable and not ignore_shear_rotate:
    #        raise ValueError("This transform does not allow 1D outputs")
    #    ds = xr.Dataset({"coord1":coord1, "coord2":coord2})
    #    stacked_array, swap_dict = xarray_utils.stack_ds_vars(ds, "spatial_var")
    #    stacked_array = stacked_array.transpose(..., "spatial_var").sortby("spatial_var")
    #    project_array = self._perform_projection(stacked_array, ignore_shear_rotate=ignore_shear_rotate)
    #    ds = xarray_utils.unstack_ds_vars(project_array, swap_dict, "spatial_var")
    #    if coord1_outname is not None:
    #        ds = ds.rename({"coord1":coord1_outname})
    #    if coord2_outname is not None:
    #        ds = ds.rename({"coord2":coord2_outname}) 
    #    return dict(ds.variables)
        
        
    def _project_seperately(self, coord1, coord2, 
           coord1_outname=None, coord2_outname=None, ignore_shear_rotate=False):
        if not self.is_seperable and not ignore_shear_rotate:
            raise ValueError("This transform does not allow 1D outputs")
        transform = self.compose_transform(self.decomposed_transform,
             ignore_shear_rotate=ignore_shear_rotate)
        coord1 = coord1*transform[0,0] + transform[0,2]
        coord2 = coord2*transform[1,1] + transform[1,2]
        if coord1_outname is None:
            coord1_outname = "coord1"
        if coord2_outname is None:
            coord2_outname = "coord2"
        return {coord1_outname:coord1, coord2_outname:coord2}
        
        
    def _project_2ds(self, coord1, coord2, coord1_outname=None, coord2_outname=None, ignore_shear_rotate=False):      
        ds = xr.Dataset({"coord1":coord1, "coord2":coord2})
        stacked_array = ds.to_array("spatial_var")
        stacked_array = stacked_array.transpose(..., "spatial_var").sortby("spatial_var")
        project_array = self._perform_projection(stacked_array, ignore_shear_rotate=ignore_shear_rotate)
        ds = project_array.to_dataset("spatial_var")
        if coord1_outname is not None:
            ds = ds.rename({"coord1":coord1_outname})
        if coord2_outname is not None:
            ds = ds.rename({"coord2":coord2_outname}) 
        return dict(ds.variables)
             
    def project_coordinates(self, coord1, coord2, coord1_outname=None, coord2_outname=None, ignore_shear_rotate=False):
        if len(coord1.dims)==1 and len(coord2.dims)==1:
            if self.is_seperable or ignore_shear_rotate:
                return self._project_seperately(coord1=coord1, 
                                             coord2=coord2,
                                             coord1_outname=coord1_outname,
                                             coord2_outname=coord2_outname,
                                             ignore_shear_rotate=ignore_shear_rotate)
        return self._project_2ds(coord1=coord1, 
                                 coord2=coord2,
                                 coord1_outname=coord1_outname,
                                 coord2_outname=coord2_outname,
                                 ignore_shear_rotate=ignore_shear_rotate)
    
def create_projected_coords(xr_obj, transform=None, 
                            name_coord1 = "xpix", name_coord2 = "ypix", 
                            name_new_coord1 ="X", name_new_coord2="Y",
                            ignore_shear_rotate=False):
    if transform is None:
        transform = xr_obj.attrs["transform"]
    if isinstance(transform, str):
        transform = xr_obj.attrs[transform]
    aph = AffineProjectionHandler(transform)
    proj_coords = aph.project_coordinates(coord1=xr_obj[name_coord1],
                                          coord2=xr_obj[name_coord2],
                                          coord1_outname = name_new_coord1,
                                          coord2_outname = name_new_coord2,
                                          ignore_shear_rotate = ignore_shear_rotate)
    return xr_obj.assign_coords(proj_coords)
    
    

def image_to_DataArray(image, include_name=False):
    ypix = np.arange(image.SampleBase64.shape[0])
    xpix = np.arange(image.SampleBase64.shape[1])
    transform = image.get_transform(global_coords=True, 
                                    mtransform=False)
    
    arr = xr.DataArray(image.SampleBase64,
                       dims=("y","x"),
                       coords={"xpix":("x", xpix), "ypix":("y", ypix), 
                              })
                              
    
    arr.attrs["TimeStamp"] = timeparser(image.TimeStamp)
    arr.attrs["transform"] = transform
    arr.attrs["Label"] = image.Label + " ({})".format(image.Tags["TraceRetrace"])
    
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
    return xr.Dataset(data_vars=dict(data_vars), attrs=data_vars[0][1].attrs.copy())


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
    ds =  xr.Dataset(chans).assign_attrs(ExportSettingsRenderedSpectra.create_attr_dict(spectrum))\
            .assign_coords(ExportSettingsRenderedSpectra.create_coord_dict(spectrum))

    return ds



def spectra_list_to_Dataset(spectra_list):
    return xr.concat([spectrum_to_Dataset(spectrum) for spectrum in spectra_list], dim="spectral_index", coords="all", data_vars="all")
