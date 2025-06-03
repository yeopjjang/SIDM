"""Module to define classes and methods that are helpful for scaleout"""

from dask.distributed import Client, PipInstall

def make_dask_client(address):
    """Create dask client that includes dependency installer"""
    dependencies = [
        "git+https://github.com/yeopjjang/SIDM.git",
    ]
    client = Client(address)
    client.register_plugin(PipInstall(packages=dependencies, pip_options=["--upgrade"]))
    return client

# """Module to define classes and methods that are helpful for scaleout"""

# from dask.distributed import Client, PipInstall
# import os

    
# def set_env(dask_worker):
#     os.environ["XCACHE_HOST"] = "xcache.cmsaf-dev.flatiron.hollandhpc.org"


# def make_dask_client(address):
#     """Create dask client that includes dependency installer"""
#     dependencies = [
#         # "git+https://github.com/btcardwell/SIDM.git",
#         "git+https://github.com/yeopjjang/SIDM.git",
#     ]
#     client = Client(address)
#     client.register_plugin(PipInstall(packages=dependencies, pip_options=["--upgrade"]))
#     client.run(set_env)
    
#     return client