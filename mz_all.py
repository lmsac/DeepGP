# a depth-first search (DFS) algorithm 
import dgl
import ipdb
import copy
# Define the graph and node-to-index mapping
edge_index = [[0, 1],[1,2], [1, 3], [3, 4], [4, 5], [4, 6]]
g = dgl.graph(edge_index)
g = dgl.to_bidirected(g)
# g = dgl.graph(([0], [1]))
# [[0, 1], [1, 2], [1, 3], [3, 4], [4, 5], [4, 6]  (N(F)(N(H(H)(H))))
# Define a DFS function to explore all possible paths
def dfs(curr_node, path, paths):
    # Add the current node to the path
    oldpath=copy.deepcopy(path)
    if curr_node not in path:
        path.append(curr_node)
        print("path",path)
    # If the path has more than one node, add it to the list of paths
    if len(path) > 1 and path != oldpath:
        print("path[:]",path[:])
        paths.append(path[:])
    print("paths",paths)
    # Recursively explore all possible neighbors of the current node
    ipdb.set_trace()
    for neighbor in  g.in_edges(curr_node)[0]:
        # # ipdb.set_trace()
        # if neighbor in path:
        #     pass
        # else:
        print("curr_node",curr_node)
        print(" g.in_edges(curr_node)[0]")
        print("neighbor",neighbor)
        dfs(neighbor, path, paths)
    # for neighbor in g.predecessors(curr_node):
    #     dfs(neighbor, path, paths)
    # for neighbor in g.in_edges(curr_node):
    #     dfs(neighbor, path, paths)
    
    # Remove the current node from the path to backtrack
    print(path)
    print("pop")
    path.pop()
#再加上单糖就可以

# Find all possible paths in the graph
paths = []
for node in g.nodes():
    # ipdb.set_trace()
    dfs(node, [], paths)
print(paths)



# Extract the compositions of fragments from the paths
# fragments = []
# for path in paths:
#     ipdb.set_trace()
    # fragment = [node2idx[g.nodes[node].data] for node in path]
    # fragment = [node2idx[g.nodes()[node].ndata['label']] for node in path]

    # fragments.append(fragment)

# Print the results
# print(fragments)









# def dfs(curr_node, path, paths, g):
#     path.append(curr_node)
#     if len(path) > 1:
#         paths.append(path[:])
#     for neighbor in g.successors(curr_node):
#         dfs(neighbor, path, paths, g)
#     for neighbor in g.predecessors(curr_node):
#         dfs(neighbor, path, paths, g)
#     path.pop()


# paths = []
# glyco_graph=glyco_process("(N(F)(N(H(H)(H))))")
# edge_index=glyco_graph[2]
# print("edge_index",edge_index)
# nodef="P"+glyco_graph[3]
# nodef=[node2idx[i] for i in nodef]
# # print("nodef",nodef)

# g=dgl.graph(edge_index)
# # g = dgl.to_bidirected(g, copy_ndata=True)
# g.ndata["mononer"]=torch.Tensor(nodef).to(int)

# # import ipdb
# # ipdb.set_trace()
# for node in g.nodes():
#     dfs(node, [], paths, g)
# print(paths)
# for i in paths:
#     ipdb.set_trace()
#     # res_monomer=g.ndata["mononer"][i].numpy().tolist()
#     res_monomer=[int(g.ndata["mononer"][index]) for index in i]
#     res_sugar=[node2dict[i] for i in res_monomer if i != 0]
#     print(res_sugar)