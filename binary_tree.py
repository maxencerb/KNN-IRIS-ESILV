import numpy as np

class Node:

    def profondeur(self):
        if self.isLeaf:
            return 1
        return 1 + max(self.left.profondeur(), self.right.profondeur())

    @staticmethod
    def parcours_largeur(node, layer, profondeur, tree = None):
        if tree is None:
            tree = [[] for _ in range(profondeur)]
        tree[layer] += [node.label]
        if not node.isLeaf:
            tree = Node.parcours_largeur(node.left, layer + 1, profondeur, tree)
            tree = Node.parcours_largeur(node.right, layer + 1, profondeur, tree)
        return tree
        


    def __str__(self):
        return '\n'.join(list(map(str, Node.parcours_largeur(self, 0, self.profondeur()))))

    def __init__(self, value, label, left = None, right = None, isLeaf = False):
        self.value = value
        self.label = label
        self.left: Node or None = left
        self.right: Node or None = right
        self.isLeaf = isLeaf