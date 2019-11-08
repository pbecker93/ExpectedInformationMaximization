from recording.Recorder import RecorderModule
import os
import json


"""Simple recording models - most of them just print information"""
class TrainIterationRecMod(RecorderModule):
    """Prints out current training iteration"""
    def record(self, iteration):
        self._logger.info("--- Iteration {:5d} ---".format(iteration))

    @property
    def logger_name(self):
        return "Iteration"


class ConfigInitialRecMod(RecorderModule):

    def record(self, name, config):
        self._logger.info((10 + len(name)) * "-")
        self._logger.info("---- " + name + " ----")
        self._logger.info((10 + len(name)) * "-")
        for k in config.keys():
            self._logger.info(str(k) + " : " + str(config[k]))
        if self._save:
            filename = os.path.join(self._save_path, "config.json")
            with open(filename, "w") as file:
                json.dump(config.__dict__, file, separators=(",\n", ": "))

    @property
    def logger_name(self):
        return "Config"
