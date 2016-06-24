from functools import wraps
import time
def time_me(function):
    @wraps(function)
    def function_timer(*args, **kwargs):
        t0 = time.time()
        result = function(*args, **kwargs)
        t1 = time.time()
        print ("Time running %s: %s seconds" %
               (function.func_name, str(t1-t0))
               )
        return result
    return function_timer
