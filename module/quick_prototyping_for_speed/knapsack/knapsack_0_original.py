from __future__ import division, print_function
import sys
from collections import namedtuple

# each item has index, value and weight properties
Item = namedtuple("Item", ['index', 'value', 'weight'])

# simple "Linearization of constraints" estimation
def estimate(items, K):
    v = 0
    for item in items:
        if item.weight > K:
            return v + item.value * K // item.weight
        K -= item.weight
        v += item.value
    return v

# assume items are sorted by density
def search(items, K, best_v=0, current_v=0, current_line=[]):    
    # collect items that could be packed into the knaspack without breaking the weight limit    
    left = [item for item in items if item.weight <= K]
    if not left:
        return current_v, current_line
    
    # calculate the "relative" goal we want to beat
    lb = best_v - current_v
    
    # estimate upperbound and test whether there is no need to try
    ub = estimate(left, K)
    if ub <= lb:
        return 0, []
    
    # recalculate the "absolute" upperbound
    ub = ub + current_v
    
    # standard search loop
    best_idxs = None
    for i, item in enumerate(left):
        v, idxs = search(left[i + 1:], K-item.weight, best_v, current_v+item.value, current_line+[item.index])
        if v > best_v:
            best_v, best_idxs = v, idxs
            print(best_v, best_idxs, "ub=", ub)
    return best_v, best_idxs

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
