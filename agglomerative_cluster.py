import json
import pickle

class node:

    def __init__(self, data=None):
        self.data = data
        self.children = []

    def add_child(self, child_node):
        self.children.append(child_node)

    def size(self):
        size = 1
        for c in self.children:
            size += c.size()
        return size

    def output_json(self):
        data = {}
        data['size'] = self.size()
        data['children'] = []
        if self.data:
            data['name'] = self.data
        for c in self.children:
            data['children'].append(c.output_json())

        return data

def generate_cost(node1, node2, pairwise_dict):
    """
    Implementing the average agglomerative method
    """
    #print(pairwise_dict)
    #print(node1.data)
    #print(node2.data)
    if node1.data != None and node2.data != None:
        return pairwise_dict[node1.data][node2.data]
    if node1.data == None:
        costs = []
        for c in node1.children:
            costs.append(generate_cost(c, node2, pairwise_dict))
        return (sum(costs)*1.0)/len(costs)
    else:
    #if node2.data == None:
        costs = []
        for c in node2.children:
            costs.append(generate_cost(c, node1, pairwise_dict))
        return (sum(costs)*1.0)/len(costs)
    #else:

def best_pair(nodes, pairwise_dict):
    max_similarity = float('-inf')
    c1 = None
    c2 = None
    costs = {}
    for n1 in nodes:
        if n1 not in costs:
            costs[n1] = {}
        for n2 in nodes:
            if n2 == n1:
                continue
            costs[n1][n2] = generate_cost(n1, n2, pairwise_dict)
            if costs[n1][n2] > max_similarity:
                max_similarity = costs[n1][n2]
                c1 = n1
                c2 = n2

    return c1,c2

def agglomerative_cluster(pairwise_dict):
    nodes = []
    for phrase in pairwise_dict:
        nodes.append(node(phrase))

    while len(nodes) > 1:
        print(len(nodes))
        n1, n2 = best_pair(nodes, pairwise_dict)
        nodes.remove(n1)
        nodes.remove(n2)
        parent = node()
        parent.add_child(n1)
        parent.add_child(n2)
        nodes.append(parent)

    print(json.dumps(nodes[0].output_json()))

if __name__ == "__main__":
    data = pickle.load(open('similarity.p', 'rb'))
    agglomerative_cluster(data)



