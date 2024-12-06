import numpy as np
import xarray as xr
import math
from collections import namedtuple

def stack_ds_vars(ds, new_dim_name=None):
    """takes xr.Dataset with multiple 1D variables and returns
    a DataArray with all vars stacked with a common index. Second output is 
    a dictionary that maps variable names to dictionaries needed to undo the stack"""
    for name, var in ds.variables.items():
        if len(var.dims)!=1:
            raise ValueError("All arrays need to be 1D")
    if len(set(ds.dims.values())) != 1:
        raise ValueError("All dimensions must have the same length")
    new_name = "index_".format("_".join(list(ds.dims.keys())))
    replace_dict = {var:{new_name:ds[var].dims[0]} for var in ds.variables.keys()}
    swap_dims = {dim:new_name for dim in ds.dims}
    return ds.swap_dims(swap_dims).to_array(new_dim_name), replace_dict
           
           
def unstack_ds_vars(array,replace_dict , new_dim_name=None):
    """takes xr.DataArray stacked by stack_ds_vars and unstacks and renames"""
    if new_dim_name is None:
        new_dim_name = "variable"
    ds = array.to_dataset(new_dim_name)
    def renamer(var):
        return var.swap_dims(replace_dict[var.name])
    
    return ds.map(renamer)







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
        transform = self.transform #self.compose_transform(self.decomposed_transform,
             #ignore_shear_rotate=ignore_shear_rotate)
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
    
