# """Module to define classes and methods that are helpful for scaleout"""

# from dask.distributed import Client, PipInstall


# def make_dask_client(address):
#     """Create dask client that includes dependency installer"""
#     dependencies = [
#         "git+https://github.com/yeopjjang/SIDM.git@coffea2025_crosscleaning",
#     ]
#     client = Client(address)
#     client.register_plugin(PipInstall(packages=dependencies, pip_options=["--upgrade"]))
#     return client

from dask.distributed import Client, PipInstall, WorkerPlugin

class CoffeaLocal2GlobalPatch(WorkerPlugin):
    def setup(self, worker):
        # 워커 프로세스에서 실행됨
        import awkward
        from coffea.nanoevents import transforms as t

        def local2global(stack):
            target_offsets = t.ensure_array(stack.pop())
            index = stack.pop()
            index = index.mask[index >= 0] + target_offsets[:-1]
            index = index.mask[index < target_offsets[1:]]
            out = t.ensure_array(
                awkward.flatten(awkward.fill_none(index, -1), axis=None)
            )
            # if out.dtype != numpy.int64:
            #     raise RuntimeError
            stack.append(out)

        # 실제 패치
        t.local2global = local2global

def make_dask_client(address):
    """Create dask client that includes dependency installer + coffea patch"""
    dependencies = [
        "git+https://github.com/yeopjjang/SIDM.git@coffea2025_crosscleaning",
    ]
    client = Client(address)
    client.register_plugin(PipInstall(packages=dependencies, pip_options=["--upgrade"]))

    # 워커가 뜰 때마다 자동 패치되도록 등록
    client.register_worker_plugin(CoffeaLocal2GlobalPatch(),
                                  name="coffea_local2global_patch")

    # (선택) 패치가 먹었는지 빠르게 검증
    def _is_patched():
        from coffea.nanoevents import transforms as t
        return "RuntimeError" not in (t.local2global.__code__.co_consts or ())

    patched = client.run(_is_patched)
    print(patched)  # 모든 워커가 True면 성공

    return client
