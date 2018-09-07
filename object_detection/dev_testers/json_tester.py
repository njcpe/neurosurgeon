import json

from collections import defaultdict


class NestedDefaultDict(defaultdict):
    def __init__(self, depth, default=int, _root=True):
        self.root = _root
        self.depth = depth
        if depth > 1:
            def cur_default(): return NestedDefaultDict(depth - 1,
                                                        default,
                                                        False)
        else:
            cur_default = default
        defaultdict.__init__(self, cur_default)

    def __repr__(self):
        if self.root:
            return "NestedDefaultDict(%d): {%s}" % (self.depth,
                                                    defaultdict.__repr__(self))
        else:
            return defaultdict.__repr__(self)


# Quick Example
def core_data_type(): return [0] * 10


test = NestedDefaultDict(3, core_data_type)
test['hello']['world']['example'][5] += 100
print(test)
print(json.dumps(test))

# Code without custom class.
test = defaultdict(lambda: defaultdict(lambda: defaultdict(core_data_type)))
test['hello']['world']['example'][5] += 100
print(test)
print(json.dumps(test))
