from objdict import ObjDict
import json


class detected_object(ObjDict):
    __keys__ = 'object_location object_type probability'


t = detected_object([0.8412602543830872, 0.2801617980003357,
                     0.9994886517524719, 0.7832360863685608], "tv", 0.74)
print(t.items())
print(json.dumps(t, indent=1, sort_keys=True))
