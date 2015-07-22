from __future__ import division, print_function
import sys
import numpy as np
try:
    import cython
    is_compiled = cython.compiled
except:
    print("can't import cython")
    is_compiled = False
    
if sys.version_info <(3,):
    range = xrange

# simple "Linearization of constraints" estimation
def estimate(items, K0):  
  v, K = 0, K0  
  for i in range(items.shape[0]):
    item = items[i]
    if item.weight > K0:
        continue
    if item.weight> K:
      return v + item.value*K//item.weight
    K-=item.weight
    v+=item.value
  return v
    
# assume items are sorted by density
def _search(items, K, best_v=0, current_v=0):    
  # see wheter ther are items that could be packed into the knaspack without breaking the weight limit
  # unfortunately, if all(item.weight>K for item in items): return current_v, won't work for cython
  for i in range(items.shape[0]):
    if items[i].weight <= K:
      break
  else:
    return current_v
  
  # estimate upperbound and test whether there is no need to try
  if current_v + estimate(items, K) <= best_v:
    return 0
  
  # standard search loop
  for i in range(items.shape[0]):
    item = items[i]
    if item.weight > K:
      continue
    v = _search(items[i+1:], K-item.weight, best_v, current_v+item.value)      
    if v > best_v:      
      best_v = v
      print(best_v, item.index)
  return best_v

# a wrapper to handle the difference between cython/typed memeoryview version and python/numpy version
def search(items, K, best_v=0, current_v=0):
    if is_compiled:
      print("compiled")
      item_list = [(item.index, item.value, item.weight) for item in items]
      _items = np.array(item_list, dtype = [('index', 'i4'), ('value', 'i4'), ('weight', 'i4')])
    else:
      print("intepreted")
      _items = np.array(items, dtype=type(items[0]))
    r = _search(_items, K, best_v, current_v)
    return r

# each item has index, value and weight properties
class Item:
    def __init__(self, i, v, w):
        self.index = i
        self.value = v
        self.weight = w


# main routine
def solve(input_filename):
    data_iter = (line.split() for line in open(input_filename))
    data = [Item(i - 1, int(v), int(w)) for i, (v, w) in enumerate(data_iter)]
    
    # get item_count and capacity from first line
    item_count, capacity = data[0].value, data[0].weight
    
    # get the value and weight of every item and sort them by density(value/weight) and using -value to break tie. 
    items = sorted((item for item in data[1:] if item.value > 0), key=lambda x: (x.weight / x.value, -x.value))
    
    # call the algorithm
    search(items, capacity)

if __name__ == "__main__":
    # read data from file
    input_filename = sys.argv[1] if len(sys.argv) > 1 else "ks_10000_0"
    solve(input_filename)