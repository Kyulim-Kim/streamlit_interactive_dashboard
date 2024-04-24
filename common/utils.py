from common.config import *
from common.env import *


lmap         = lambda fn, arr: list(map(fn, arr))
inverse_dict = lambda dic: {value: key for key, value in dic.items()}
