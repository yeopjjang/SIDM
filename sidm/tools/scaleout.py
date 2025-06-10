"""Module to define classes and methods that are helpful for scaleout"""

from dask.distributed import Client, PipInstall

def make_dask_client(address):
    """Create dask client that includes dependency installer"""
    dependencies = [
        "git+https://github.com/yeopjjang/SIDM.git@LJ_Isolation_summary",
    ]
    client = Client(address)
    client.register_plugin(PipInstall(packages=dependencies, pip_options=["--upgrade"]))\
    
    return client

