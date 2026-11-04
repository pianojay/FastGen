"""Minimal attribute dictionary for pinned upstream API; no external dependency."""
class Munch(dict):
    def __getattr__(self,key):
        try:return self[key]
        except KeyError as e:raise AttributeError(key) from e
    def __setattr__(self,key,value):self[key]=value
