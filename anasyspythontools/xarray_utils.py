

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
        return var.swap_dims(replace_dict[x.name])
    
    return ds.map(rename)
           
