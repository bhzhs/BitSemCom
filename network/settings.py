# settings.py
class GlobalConfig:
    instance = None

    @classmethod
    def set_config(cls, config):
        cls.instance = config

    @classmethod
    def get_config(cls):
        if cls.instance is None:
            raise ValueError("GlobalConfig is not set.")
        return cls.instance
