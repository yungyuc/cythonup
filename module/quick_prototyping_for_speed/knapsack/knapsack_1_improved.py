from __future__ import division, print_function
import sys
from collections import namedtuple

if sys.version_info <(3,):
    range = xrange

# each item has index, value and weight properties
Item = namedtuple("Item", ['index', 'value', 'weight'])

# simple "Linearization of constraints" estimation
def estimate(items, K0):
  v, K = 0, K0  
  for item in items:
    if item.weight > K0:
        continue
    if item.weight> K:
      return v + item.value*K//item.weight
    K-=item.weight
    v+=item.value
  return v
    
# assume items are sorted by density
def search(items, K, best_v=0, current_v=0):  
  # see wheter ther are items that could be packed into the knaspack without breaking the weight limit
  # unfortunately, if all(item.weight>K for item in items): return current_v, won't work for cython
  for item in items:
    if item.weight <= K:      
      break
  else:
    return current_v

  # estimate upperbound and test whether there is no need to try
  if current_v + estimate(items, K) <= best_v:
    return 0

  # standard search loop
  for i in range(len(items)):
      item = items[i]
      if item.weight > K:
        continue
      v = search(items[i+1:], K-item.weight, best_v, current_v+item.value)
      if v > best_v:      
          best_v = v
          print(best_v, item.index)
  return best_v

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
