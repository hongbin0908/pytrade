"""
find a way to print the decision path of tree
"""

import os
import sys
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.tree import _tree
from sklearn.tree import _criterion
from sklearn.externals import six
local_path = os.path.dirname(__file__)

LEFT = 1
RIGHT = 2

def export_decision_path(decision_tree, out_file=None, feature_names=None,
                         label='all',
                         special_characters=False,
                         node_ids = False,
                         rounded = True,
                         proportion = False,
                         impurity = True,
                         class_names=None):
    def recurse(the_tree, node_id):
        if node_id == _tree.TREE_LEAF:
            raise ValueError("Invalid node_id %s" % _tree.TREE_LEAF)
        left_child = the_tree.children_left[node_id]
        right_child = the_tree.children_right[node_id]
        if left_child != _tree.TREE_LEAF:
            child2parent[left_child] = (LEFT, node_id)
            child2parent[right_child] = (RIGHT, node_id)
            recurse(the_tree, left_child)
            recurse(the_tree, right_child)
        else:
            leafs.append(node_id)

    def node_to_str(tree, node, criterion):
        node_id = node[1]
        node_pos = node[0]
        # Generate the node content string
        if tree.n_outputs == 1:
            value = tree.value[node_id][0, :]
        else:
            value = tree.value[node_id]

        # Should labels be shown?
        labels = (label == 'root' and node_id == 0) or label == 'all'

        # PostScript compatibility for special characters
        if special_characters:
            characters = ['&#35;', '<SUB>', '</SUB>', '&le;', '<br/>', '>']
            node_string = '<'
        else:
            characters = ['#', '[', ']', '<=', '\\n', '"', '>']
            node_string = '"'

        # Write node ID
        if node_ids:
            if labels:
                node_string += 'node '
            node_string += characters[0] + str(node_id) + characters[4]

        # Write decision criteria
        if tree.children_left[node_id] != _tree.TREE_LEAF:
            # Always write node decision criteria, except for leaves
            if feature_names is not None:
                feature = feature_names[tree.feature[node_id]]
            else:
                feature = "X%s%s%s" % (characters[1],
                                       tree.feature[node_id],
                                       characters[2])
            node_string += '%s %s %s%s' % (feature,
                                           characters[3] if node_pos == 1 else characters[6],
                                           round(tree.threshold[node_id], 4),
                                           characters[4])

        # Write impurity
        if impurity:
            if isinstance(criterion, _criterion.FriedmanMSE):
                criterion = "friedman_mse"
            elif not isinstance(criterion, six.string_types):
                criterion = "impurity"
            if labels:
                node_string += '%s = ' % criterion
            node_string += (str(round(tree.impurity[node_id], 4)) +
                            characters[4])

        # Write node sample count
        if labels:
            node_string += 'samples = '
        if proportion:
            percent = (100. * tree.n_node_samples[node_id] /
                       float(tree.n_node_samples[0]))
            node_string += (str(round(percent, 1)) + '%' +
                            characters[4])
        else:
            node_string += (str(tree.n_node_samples[node_id]) +
                            characters[4])

        # Write node class distribution / regression value
        if proportion and tree.n_classes[0] != 1:
            # For classification this will show the proportion of samples
            value = value / tree.weighted_n_node_samples[node_id]
        if labels:
            node_string += 'value = '
        if tree.n_classes[0] == 1:
            # Regression
            value_text = np.around(value, 4)
        elif proportion:
            # Classification
            value_text = np.around(value, 2)
        elif np.all(np.equal(np.mod(value, 1), 0)):
            # Classification without floating-point weights
            value_text = value.astype(int)
        else:
            # Classification with floating-point weights
            value_text = np.around(value, 4)
        # Strip whitespace
        value_text = str(value_text.astype('S32')).replace("b'", "'")
        value_text = value_text.replace("' '", ", ").replace("'", "")
        if tree.n_classes[0] == 1 and tree.n_outputs == 1:
            value_text = value_text.replace("[", "").replace("]", "")
        value_text = value_text.replace("\n ", characters[4])
        node_string += value_text + characters[4]

        # Write node majority class
        if (class_names is not None and
                    tree.n_classes[0] != 1 and
                    tree.n_outputs == 1):
            # Only done for single-output classification trees
            if labels:
                node_string += 'class = '
            if class_names is not True:
                class_name = class_names[np.argmax(value)]
            else:
                class_name = "y%s%s%s" % (characters[1],
                                          np.argmax(value),
                                          characters[2])
            node_string += class_name

        # Clean up any trailing newlines
        if node_string[-2:] == '\\n':
            node_string = node_string[:-2]
        if node_string[-5:] == '<br/>':
            node_string = node_string[:-5]

        return node_string + characters[5]

    # open out file
    return_string = False
    own_file = False
    if isinstance(out_file, six.string_types):
        if six.PY3:
            out_file = open(out_file, "w", encoding="utf-8")
        else:
            out_file = open(out_file, "wb")
        own_file = True
    if out_file is None:
        return_string = True
        out_file = six.StringIO()

    out_file.write('digraph Decision_path {\n')
    out_file.write('rankdir = LR;\n')
    out_file.write('node [shape=box];\n')

    child2parent = {}
    leafs = []
    recurse(decision_tree.tree_, 0)

    idx = 0
    for leaf in leafs:
        path = []
        cur_node = (0, leaf)
        while True:
            path.append(cur_node)
            if cur_node[1] == 0:
                break
            cur_node = child2parent[cur_node[1]]
        path.reverse()
        for node in path:
            out_file.write('f%dt%d [label=%s];\n'
                           % (idx, node[1], node_to_str(decision_tree.tree_, node, decision_tree.criterion)))
            if node[1] != 0:
                out_file.write('f%dt%d -> f%dt%d;\n' % (idx, child2parent[node[1]][1], idx, node[1]))
        idx += 1
    out_file.write('}')

    if return_string:
        return out_file.getvalue()
    if own_file:
        out_file.close()

def export_decision_path2(random_tree, x, out_file=None, feature_names=None,
                         label='all',
                         special_characters=False,
                         node_ids = False,
                         rounded = True,
                         proportion = False,
                         impurity = True,
                         class_names=None):
    def recurse(the_tree, node_id):
        if node_id == _tree.TREE_LEAF:
            raise ValueError("Invalid node_id %s" % _tree.TREE_LEAF)
        left_child = the_tree.children_left[node_id]
        right_child = the_tree.children_right[node_id]
        if left_child != _tree.TREE_LEAF:
            child2parent[left_child] = (LEFT, node_id)
            child2parent[right_child] = (RIGHT, node_id)
            recurse(the_tree, left_child)
            recurse(the_tree, right_child)
        else:
            leafs.append(node_id)

    def node_to_str(tree, node, criterion):
        node_id = node[1]
        node_pos = node[0]
        # Generate the node content string
        if tree.n_outputs == 1:
            value = tree.value[node_id][0, :]
        else:
            value = tree.value[node_id]

        # Should labels be shown?
        labels = (label == 'root' and node_id == 0) or label == 'all'

        # PostScript compatibility for special characters
        if special_characters:
            characters = ['&#35;', '<SUB>', '</SUB>', '&le;', '<br/>', '>']
            node_string = '<'
        else:
            characters = ['#', '[', ']', '<=', '\\n', '"', '>']
            node_string = '"'

        # Write node ID
        if node_ids:
            if labels:
                node_string += 'node '
            node_string += characters[0] + str(node_id) + characters[4]

        # Write decision criteria
        if tree.children_left[node_id] != _tree.TREE_LEAF:
            # Always write node decision criteria, except for leaves
            if feature_names is not None:
                feature = feature_names[tree.feature[node_id]]
            else:
                feature = "X%s%s%s" % (characters[1],
                                       tree.feature[node_id],
                                       characters[2])
            node_string += '%s %s %s%s' % (feature,
                                           characters[3] if node_pos == 1 else characters[6],
                                           round(tree.threshold[node_id], 4),
                                           characters[4])

        # Write impurity
        if impurity:
            if isinstance(criterion, _criterion.FriedmanMSE):
                criterion = "friedman_mse"
            elif not isinstance(criterion, six.string_types):
                criterion = "impurity"
            if labels:
                node_string += '%s = ' % criterion
            node_string += (str(round(tree.impurity[node_id], 4)) +
                            characters[4])

        # Write node sample count
        if labels:
            node_string += 'samples = '
        if proportion:
            percent = (100. * tree.n_node_samples[node_id] /
                       float(tree.n_node_samples[0]))
            node_string += (str(round(percent, 1)) + '%' +
                            characters[4])
        else:
            node_string += (str(tree.n_node_samples[node_id]) +
                            characters[4])

        # Write node class distribution / regression value
        if proportion and tree.n_classes[0] != 1:
            # For classification this will show the proportion of samples
            value = value / tree.weighted_n_node_samples[node_id]
        if labels:
            node_string += 'value = '
        if tree.n_classes[0] == 1:
            # Regression
            value_text = np.around(value, 4)
        elif proportion:
            # Classification
            value_text = np.around(value, 2)
        elif np.all(np.equal(np.mod(value, 1), 0)):
            # Classification without floating-point weights
            value_text = value.astype(int)
        else:
            # Classification with floating-point weights
            value_text = np.around(value, 4)
        # Strip whitespace
        value_text = str(value_text.astype('S32')).replace("b'", "'")
        value_text = value_text.replace("' '", ", ").replace("'", "")
        if tree.n_classes[0] == 1 and tree.n_outputs == 1:
            value_text = value_text.replace("[", "").replace("]", "")
        value_text = value_text.replace("\n ", characters[4])
        node_string += value_text + characters[4]

        # Write node majority class
        if (class_names is not None and
                    tree.n_classes[0] != 1 and
                    tree.n_outputs == 1):
            # Only done for single-output classification trees
            if labels:
                node_string += 'class = '
            if class_names is not True:
                class_name = class_names[np.argmax(value)]
            else:
                class_name = "y%s%s%s" % (characters[1],
                                          np.argmax(value),
                                          characters[2])
            node_string += class_name

        # Clean up any trailing newlines
        if node_string[-2:] == '\\n':
            node_string = node_string[:-2]
        if node_string[-5:] == '<br/>':
            node_string = node_string[:-5]

        return node_string + characters[5]

    # open out file
    return_string = False
    own_file = False
    if isinstance(out_file, six.string_types):
        if six.PY3:
            out_file = open(out_file, "w", encoding="utf-8")
        else:
            out_file = open(out_file, "wb")
        own_file = True
    if out_file is None:
        return_string = True
        out_file = six.StringIO()

    out_file.write('digraph Decision_path {\n')
    out_file.write('rankdir = LR;\n')
    out_file.write('node [shape=box];\n')

    scores = []
    for tree in random_tree.estimators_:
        scores.append((tree, float(tree.predict_proba(x)[:,1])))
    iter = 0
    for each in sorted(scores, key = lambda x : x[1], reverse = True)[0:10]:
        child2parent = {}
        leafs = []
        recurse(each[0].tree_, 0)

        leaf_of_path = -1
        path = each[0].decision_path(x)[0].todense()[0,:].tolist()[0]
        print(path)
        idx = len(path) - 1
        while True:
            if path[idx] == 1:
                leaf_of_path = idx
                break
            idx -= 1

        for leaf in leafs:
            if leaf != leaf_of_path:
                idx += 1
                continue
            path = []
            cur_node = (0, leaf)
            while True:
                path.append(cur_node)
                if cur_node[1] == 0:
                    break
                cur_node = child2parent[cur_node[1]]
            path.reverse()
            for node in path:
                out_file.write('f%dt%d [label=%s];\n'
                               % (iter, node[1], node_to_str(each[0].tree_, node, each[0].criterion)))
                if node[1] != 0:
                    out_file.write('f%dt%d -> f%dt%d;\n' % (iter, child2parent[node[1]][1], iter, node[1]))
        iter += 1
    out_file.write('}')

    if return_string:
        return out_file.getvalue()
    if own_file:
        out_file.close()
#set_bc = load_breast_cancer()
#X = set_bc["data"]
#y = set_bc["target"]
#assert isinstance(X, np.ndarray)
#assert isinstance(y, np.ndarray)
#
## classfier = DecisionTreeClassifier(max_depth=4)
#classfier = RandomForestClassifier(min_samples_leaf=100, n_estimators=100)
#classfier.fit(X, y)
## >>>print(type(classfier.decision_path(X)[0]))
## <class 'scipy.sparse.csr.csr_matrix'>
##print(classfier.decision_path(X)[0].todense()[0])
#np_dpth = classfier.estimators_[0].decision_path(X[0])[0].todense()
## export_decision_path2(classfier.estimators_[0], np_dpth[0,:].tolist()[0], os.path.join(local_path, 'decision_path.dot'))
#export_decision_path2(classfier, X[0], os.path.join(local_path, 'decision_path.dot'))

#print(classfier.estimators_[0])
#tree.export_graphviz(classfier, sys.stdout)
#export_decision_path(classfier.estimators_[0], os.path.join(local_path, 'decision_path.dot'))

