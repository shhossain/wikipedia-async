import logging


class CustomLogger(logging.Logger):
    def __init__(self, name, level=logging.NOTSET):
        super().__init__(name, level)
        self._logged_messages = set()

    def once(self, level, msg, *args, **kwargs):
        """Log message only once per process run"""
        key = (level, msg % args if args else msg)
        if key in self._logged_messages:
            return
        self._logged_messages.add(key)
        self.log(level, msg, *args, **kwargs)


logger = CustomLogger("wikipedia_async")
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
logger.addHandler(handler)
