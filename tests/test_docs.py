import iq_readout


def test_docstring_existence():
    def scan_module(module):
        if module.__doc__ is None:
            raise ValueError(f"{module} has no docstring")

        if "__all__" in dir(module):
            for submodule_name in module.__all__:
                submodule = getattr(module, submodule_name)
                scan_module(submodule)

        return

    scan_module(iq_readout)

    return
