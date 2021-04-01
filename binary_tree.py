class Node:
    """
    Simple Binary Tree data structure adapted for k-nn algorithm

    Arguments
    ---------
    value:  Any

    label:  Any

    left:   Node or None (left child)

    right:  Node or None (right child)

    isLeaf: boolean to force leaf

    Methods
    -------
    profondeur: gets the height of the binary tree


    Usage
    -----
    ```python
    tree = Node(1, 'top')
    left = Node(2, 'bottom', isLeaf = True)
    right = Node(3, 'bottom-right', isLeaf = True)
    tree.left = left
    tree.right = right
    print(tree)
    # output :
    # top
    # bottom bottom-right
    ```
    """

    def profondeur(self):
        """
        Gets the height of the current structure

        Usage
        -----
        ```python
        tree = Node(1, 'top')
        left = Node(2, 'left', isLeaf = True)
        child = Node(3, 'bottom', isLeaf = True)
        tree.left = left
        left.left = child
        print(tree)
        # output :
        # top
        # left 
        # bottom
        print(tree.profondeur())
        # Output :
        # 3
        ```
        """
        if self.isLeaf:
            return 1
        return 1 + max(self.left.profondeur(), self.right.profondeur())

    @staticmethod
    def __parcours_largeur(node, layer, profondeur, tree = None):
        if tree is None:
            tree = [[] for _ in range(profondeur)]
        tree[layer] += [node.label]
        if not node.isLeaf:
            tree = Node.__parcours_largeur(node.left, layer + 1, profondeur, tree)
            tree = Node.__parcours_largeur(node.right, layer + 1, profondeur, tree)
        return tree

    def __str__(self):
        return '\n'.join(list(map(str, Node.__parcours_largeur(self, 0, self.profondeur()))))

    def __init__(self, value, label, left = None, right = None, isLeaf = False):
        self.value = value
        self.label = label
        self.left: Node or None = left
        self.right: Node or None = right
        self.isLeaf = isLeaf