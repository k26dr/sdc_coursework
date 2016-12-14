from miniflow import *

x, y, z = Input(), Input(), Input()
f = Add(x,y,z)

feed_dict = { x: 10, y: 20, z: 30 }

sorted_neurons = topological_sort(feed_dict)
output = forward_pass(f, sorted_neurons)

print(output)
