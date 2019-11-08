import warnings


class ConfigDict:

    def __init__(self, **kwargs):
        self._adding_permitted = True
        self._modifying_permitted = True
        self._c_dict = {**kwargs}
        self._initialized = True

    def __setattr__(self, key, value):
        if "_initialized" in self.__dict__:
            if self._adding_permitted:
                self._c_dict[key] = value
            else:
                if self._modifying_permitted and key in self._c_dict.keys():
                    self._c_dict[key] = value
                elif key in self._c_dict.keys():
                    raise AssertionError("Tried modifying existing parameter after modifying finalized")
                else:
                    raise AssertionError("Tried to add parameter after adding finalized")
        else:
            self.__dict__[key] = value

    def __getattr__(self, item):
        if "_initialized" in self.__dict__ and item in self._c_dict.keys():
            return self._c_dict[item]
        else:
            raise AssertionError("Tried accessing non existing parameter")

    def __getitem__(self, item):
        return self._c_dict[item]

    @property
    def adding_permitted(self):
        return self.__dict__["_adding_permitted"]

    @property
    def modifying_permitted(self):
        return self.__dict__["_modifying_permitted"]

    def finalize_adding(self):
        self.__dict__["_adding_permitted"] = False

    def finalize_modifying(self):
        if self.__dict__["_adding_permitted"]:
            warnings.warn("ConfigDict.finalize_modifying called while adding still allowed - also deactivating adding!")
            self.__dict__["_adding_permitted"] = False
        self.__dict__["_modifying_permitted"] = False

    def keys(self):
        return self._c_dict.keys()


if __name__ == "__main__":

    c = ConfigDict()
    c.bla = 3
    c.finalize_adding()
    c.bla = 4

    try:
        c.bla2 = 2            # invalid adding
    except AssertionError as e:
        print(e)

    c.finalize_modifying()

    try:
        c.bla = 5              # invalid modification
    except AssertionError as e:
        print(e)

    try:
        print(c.fuu)              # invalid read
    except AssertionError as e:
        print(e)
    print(c.bla)