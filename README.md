Faster Heuristics for Graph Burning
------------------------------------------

Let G(V, E) be a graph where V is a set of nodes depicting people, E
denotes a relationship among the nodes. Initially, all nodes are in an unburned
(uninformed) state. Given time steps t 0 , t 1 , t 2 · · · t b−1 , at t 0 one node is set to
fire from outside. It starts burning and spreads the fire to its neighbors in a
step-wise fashion. During the process of burning, it is assumed that either the
node is set on fire directly, called the source of fire or the node is burning by
catching fire from a neighbor or it is not yet burnt. At i th time step, a new
unburned node is set on fire from outside and all those nodes which have caught
fire at t i−1 , burn their neighbors. The process stops when the entire graph is
burning, that is, or all the nodes have received information.
Minimum number of steps to be required for burning process is called burning number.

How To run the code
------------------------------------------
"DATA/TestDATA/*.txt" path directory of files edgelist data set.

python3 file_name 
