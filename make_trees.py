from scipy.cluster.hierarchy import to_tree

def get_newick(node, parent_dist, leaf_names, newick, format_branch_length=True):
    # create newick string from a tree
    if node.is_leaf():
        return "%s:%.2f%s" % (leaf_names[node.id], parent_dist - node.dist, newick)
    else:
        if len(newick) > 0:
            newick = "):%.2f%s" % (parent_dist - node.dist, newick)
        else:
            newick = ");"

        newick = get_newick(node.get_left(), node.dist, leaf_names, newick, format_branch_length)
        newick = get_newick(node.get_right(), node.dist, leaf_names, ",%s" % (newick), format_branch_length)
        newick = "(%s" % (newick)
        return newick
    
def linkage_to_newick(Z, leaf_names):
    # convert linkage matrix to a tree and then to newick format
    tree = to_tree(Z, rd=False)
    return get_newick(tree, tree.dist, leaf_names, "")

def consensus_tree(linkage_matrices, leaf_names_list, family_names_list):
    # create a consensus tree from multiple linkage matrices
    newick_strs = [linkage_to_newick(linkage_matrix, leaf_names) for linkage_matrix, leaf_names in zip(linkage_matrices, leaf_names_list)]
    combined_newick_str = "\n".join(newick_strs)
    # write the consensus tree to a file
    with open('plots/consensus_tree.nwk', 'w') as f:
            f.write(combined_newick_str)
    print("Wrote consensus tree to file: plots/consensus_tree.nwk.")

def language_family_trees(linkage_matrices, leaf_names_list, family_names_list):
    # create trees for each language family
    newick_strs = [linkage_to_newick(linkage_matrix, leaf_names) for linkage_matrix, leaf_names in zip(linkage_matrices, leaf_names_list)]
    combined_newick_str = "\n".join(newick_strs)
    # write the language family trees to a file
    with open('plots/language_family_trees.nwk', 'w') as f:
            f.write(combined_newick_str)
    print("Wrote language family trees to file: plots/language_family_trees.nwk.")
def single_tree(data):
    # create a single tree from a linkage matrix and leaf names
    newick_str = linkage_to_newick(data[0], data[1])
    # write the single tree to a file
    with open('plots/dendrogram.nwk', 'w') as f:
        f.write(newick_str)
    print("Wrote dendrogram to file: plots/dendrogram.nwk.")

def neighbornet(matrix, labels):
    # create a nexus string from a distance matrix and labels for running in SplitsTree
    nexus_str = "#NEXUS\n\nBEGIN Taxa;\n\tDIMENSIONS ntax=%d;\n\tTAXLABELS\n" % len(matrix)
    for i, label in enumerate(labels):
        nexus_str += "\t[%d] '%s'\n" % (i + 1, label)
    nexus_str += "\t;\nEND;\n\nBEGIN Distances;\n\tDIMENSIONS ntax=%d;\n\tFORMAT labels=left diagonal triangle=both;\n\tMATRIX\n" % len(matrix)
    
    for i, label in enumerate(labels):
        nexus_str += "\t%s " % label
        for j in range(len(matrix)):
            nexus_str += "%.5f " % matrix[i][j]
        nexus_str += "\n"
    
    nexus_str += "\t;\nEND;\n"

    # write nexus string to file
    with open('plots/neighbornet.nex', 'w') as f:
        f.write(nexus_str)
    print("wrote neighbornet nexus file: plots/neighbornet.nex.")

if __name__ == "__main__":
    print("This is a module. Please run the main script.")
    pass