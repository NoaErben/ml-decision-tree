import numpy as np
import matplotlib.pyplot as plt
import queue

### Chi square table values ###
# The first key is the degree of freedom 
# The second key is the p-value cut-off
# The values are the chi-statistic that you need to use in the pruning

chi_table = {1: {0.5 : 0.45,
             0.25 : 1.32,
             0.1 : 2.71,
             0.05 : 3.84,
             0.0001 : 100000},
         2: {0.5 : 1.39,
             0.25 : 2.77,
             0.1 : 4.60,
             0.05 : 5.99,
             0.0001 : 100000},
         3: {0.5 : 2.37,
             0.25 : 4.11,
             0.1 : 6.25,
             0.05 : 7.82,
             0.0001 : 100000},
         4: {0.5 : 3.36,
             0.25 : 5.38,
             0.1 : 7.78,
             0.05 : 9.49,
             0.0001 : 100000},
         5: {0.5 : 4.35,
             0.25 : 6.63,
             0.1 : 9.24,
             0.05 : 11.07,
             0.0001 : 100000},
         6: {0.5 : 5.35,
             0.25 : 7.84,
             0.1 : 10.64,
             0.05 : 12.59,
             0.0001 : 100000},
         7: {0.5 : 6.35,
             0.25 : 9.04,
             0.1 : 12.01,
             0.05 : 14.07,
             0.0001 : 100000},
         8: {0.5 : 7.34,
             0.25 : 10.22,
             0.1 : 13.36,
             0.05 : 15.51,
             0.0001 : 100000},
         9: {0.5 : 8.34,
             0.25 : 11.39,
             0.1 : 14.68,
             0.05 : 16.92,
             0.0001 : 100000},
         10: {0.5 : 9.34,
              0.25 : 12.55,
              0.1 : 15.99,
              0.05 : 18.31,
              0.0001 : 100000},
         11: {0.5 : 10.34,
              0.25 : 13.7,
              0.1 : 17.27,
              0.05 : 19.68,
              0.0001 : 100000}}

def calc_gini(data):
    """
    Calculate gini impurity measure of a dataset.
 
    Input:
    - data: any dataset where the last column holds the labels.
 
    Returns:
    - gini: The gini impurity value.
    """
    gini = 0.0
    ###########################################################################
    # Retrieve the label column
    label_col = data[:, -1]
    label_col_len = len(label_col)
    unique, counts = np.unique(label_col, return_counts=True)
    # Divide each element by a the label's length
    probability = counts / label_col_len
    # Raise each item to the power of 2
    power_probability = np.power(probability, 2)
    # Calculate the gini impurity value
    gini = 1- np.sum(power_probability)                                       
    ###########################################################################
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return gini

def calc_entropy(data):
    """
    Calculate the entropy of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns:
    - entropy: The entropy value.
    """
    entropy = 0.0
    ###########################################################################
    # Retrieve the label column
    label_col = data[:, -1]
    label_col_len = len(label_col)
    unique, counts = np.unique(label_col, return_counts=True)
    # Divide each element by a the label's length
    probability = counts / label_col_len
    # Calculate the dot product of the probability array and the logarithm base 2 of the probability array
    log_probability = np.dot(probability,np.log2(probability))
    entropy = -np.sum(log_probability)                                       
    ###########################################################################
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return entropy

def label_count_dictionary(labels):
    """
    Counts the occurrences of each unique label in the given array of labels.
    This function creates a dictionary where the keys represent unique labels
    and the values represent the number of times each label appears in the array.

    Input:
    - labels: An array of labels.

    Returns:
    - label_counts: A dictionary where keys are unique labels and values are
    the number of occurrences of each label.
    """
    # Initialize an empty dictionary to store label counts
    label_counts = {}

    # Loop through each unique label
    for label in np.unique(labels):
        # Count the occurrences of the current label in the array of labels
        count = np.sum(labels == label)
        # Store the count in the dictionary with the label as key
        label_counts[label] = count

    return label_counts


class DecisionNode:

    
    def __init__(self, data, impurity_func, feature=-1,depth=0, chi=1, max_depth=1000, gain_ratio=False):
        
        self.data = data # the relevant data for the node
        self.feature = feature # column index of criteria being tested
        self.pred = self.calc_node_pred() # the prediction of the node
        self.depth = depth # the current depth of the node
        self.children = [] # array that holds this nodes children
        self.children_values = []
        self.terminal = False # determines if the node is a leaf
        self.chi = chi 
        self.max_depth = max_depth # the maximum allowed depth of the tree
        self.impurity_func = impurity_func
        self.gain_ratio = gain_ratio
        self.feature_importance = 0
    
    def calc_node_pred(self):
        """
        Calculate the node prediction.

        Returns:
        - pred: the prediction of the node
        """
        pred = None
        ###########################################################################
        # Retrieve the label column
        label_col = self.data[:, -1]
        unique, counts = np.unique(label_col, return_counts=True)
        # Create a dictionary from unique elements and their counts
        result_dict = dict(zip(unique, counts))
        # Find the key corresponding to the maximum value
        pred = max(result_dict, key=result_dict.get)
        ###########################################################################
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return pred
        
    def add_child(self, node, val):
        """
        Adds a child node to self.children and updates self.children_values

        This function has no return value
        """
        ###########################################################################
        self.children.append(node)
        self.children_values.append(val)
        ###########################################################################
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        
    def calc_feature_importance(self, n_total_sample):
        """
        Calculate the selected feature importance.
        
        Input:
        - n_total_sample: the number of samples in the dataset.

        This function has no return value - it stores the feature importance in 
        self.feature_importance
        """
        total_rows = len(self.data) 
        # Counter to track the number of rows counted so far
        total_rows_counted = 0 
        weighted_impurity_sum = 0  
        # Sort the dataset by the specified feature column
        feature_data_sorted = self.data[self.data[:, self.feature].argsort()]
        # Count the number of unique elements in the specified feature column
        feature_values, feature_counts = np.unique(self.data[:, self.feature], return_counts=True)
        # Compute the impurity of the entire dataset
        impurity_data = self.impurity_func(self.data)
        # Iterate over each count of unique values in the feature data
        for value, num in zip(feature_values, feature_counts):
            weighted_impurity_sum += (num / total_rows) * self.impurity_func(feature_data_sorted[total_rows_counted:total_rows_counted + num])
            total_rows_counted += num
        # Compute the goodness of split
        goodness =  impurity_data - weighted_impurity_sum
        node_probability = total_rows / n_total_sample
        # Compute the feature importance
        self.feature_importance = goodness * node_probability
        ###########################################################################
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
    
    def goodness_of_split(self, feature):
        """
        Calculate the goodness of split of a dataset given a feature and impurity function.

        Input:
        - feature: the feature index the split is being evaluated according to.

        Returns:
        - goodness: the goodness of split
        - groups: a dictionary holding the data after splitting 
                  according to the feature values.
        """
        goodness = 0
        groups = {} # groups[feature_value] = data_subset
        ###########################################################################
        total_rows = len(self.data) 
        # Counter to track the number of rows counted so far
        total_rows_counted = 0 
        weighted_impurity_sum = 0  
        split_information = 0.0  

        # Sort the dataset by the specified feature column
        feature_data_sorted = self.data[self.data[:, feature].argsort()]
        # Count the number of unique elements in the specified feature column
        feature_values, feature_counts = np.unique(self.data[:, feature], return_counts=True)

        impurity_func = self.impurity_func

        if self.gain_ratio:
            impurity_func = calc_entropy

        # Compute the impurity of the entire dataset
        impurity_data = impurity_func(self.data)

        # Iterate over each count of unique values in the feature data
        for value, num in zip(feature_values, feature_counts):
            weighted_impurity_sum += (num / total_rows) * impurity_func(feature_data_sorted[total_rows_counted:total_rows_counted + num])
            split_information -= (num / total_rows) * np.log2(num / total_rows)
            groups[value] = feature_data_sorted[total_rows_counted:total_rows_counted + num]
            total_rows_counted += num

        if self.gain_ratio:
        # Compute the information gain
            information_gain = impurity_data - weighted_impurity_sum
            if split_information==0:
                return 0, groups
            goodness = information_gain / split_information
        else:
        # Compute the goodness of split
            goodness =  impurity_data - weighted_impurity_sum

        ###########################################################################
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return goodness, groups
    
    def split(self):
        """
        Splits the current node according to the self.impurity_func. This function finds
        the best feature to split according to and create the corresponding children.
        This function should support pruning according to self.chi and self.max_depth.

        This function has no return value
        """
        ###########################################################################

        # Check if the maximum depth has been reached
        if self.depth == self.max_depth:
            self.terminal = True
            return

        # Finding the best feature to split
        best_feature_goodness = 0
        best_feature_index = -1
        # Going over all the features and calculate the goodness of split by those features
        for feature_index in range(len(self.data[0]) - 1):
            current_feature_goodness, _ = self.goodness_of_split(feature_index)
            # Finding the index of the feature that gives the highest goodness of split
            if current_feature_goodness > best_feature_goodness:
                best_feature_goodness = current_feature_goodness
                best_feature_index = feature_index

        if best_feature_index == -1:
            self.terminal = True
            return
        self.feature = best_feature_index
        _, group = self.goodness_of_split(best_feature_index)

        # Check if chi pruning condition is met
        chi_pruning_condition_met = False
        # Check if the chi value is 1 then no need to perform chi pruning
        if self.chi == 1:
            chi_pruning_condition_met = True
        else:
            # Calculate the chi value by the formula and compare it with the value from the chi table
            chi_square = 0
            label_size = len(self.data[:, -1])
            dict_label_count = label_count_dictionary(self.data[:, -1])
            for feature_val, feature_val_data in group.items():
                feature_val_data_size = len(feature_val_data)
                sub_label_count = label_count_dictionary(feature_val_data[:, -1])
                for label, count in dict_label_count.items():
                    # Calculate the parameters in the formula
                    expected = feature_val_data_size * (count / label_size)
                    observed = 0
                    if sub_label_count.get(label) is not None:
                        observed = sub_label_count.get(label)
                    # Calculate the chi square according to the formula
                    chi_square += (((observed - expected) ** 2) / expected)
            # To obtain the degrees of freedom, count feature's distinct values
            # and subtract 1, representing the variability within that feature
            if group is not None:
                deg_of_freedom = len(group) - 1
            else:
                deg_of_freedom = 0  

            if deg_of_freedom <= 0:
                self.terminal = True
                return

            chi_val_from_table = chi_table[deg_of_freedom][self.chi]
            if chi_square >= chi_val_from_table:
                chi_pruning_condition_met = True

        # Checking if the condition of chi pruning exists and if we will have more than 1 child
        if chi_pruning_condition_met:
            for key, sub_group in group.items():
                child = DecisionNode(sub_group, impurity_func=self.impurity_func, depth=self.depth + 1,
                                    chi=self.chi, max_depth=self.max_depth, gain_ratio=self.gain_ratio)
                self.add_child(child, key)
            return
        else:
            self.terminal = True
            return
        ###########################################################################
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
    

                    
class DecisionTree:
    def __init__(self, data, impurity_func, feature=-1, chi=1, max_depth=1000, gain_ratio=False):
        self.data = data # the relevant data for the tree
        self.impurity_func = impurity_func # the impurity function to be used in the tree
        self.chi = chi 
        self.max_depth = max_depth # the maximum allowed depth of the tree
        self.gain_ratio = gain_ratio #
        self.root = None # the root node of the tree
        
    def build_tree(self):
        """
        Build a tree using the given impurity measure and training dataset. 
        You are required to fully grow the tree until all leaves are pure 
        or the goodness of split is 0.

        This function has no return value
        """
        self.root = None
        ###########################################################################
        # Create the root node
        self.root = DecisionNode(data=self.data,impurity_func=self.impurity_func, chi=self.chi, max_depth=self.max_depth, gain_ratio=self.gain_ratio)

        # Use a queue to process nodes level by level
        level_queue  = queue.Queue()
        level_queue.put(self.root)

        # Loop through nodes in the queue until it's empty
        while not level_queue.empty():
            current_node = level_queue.get()

            # Check if the data at this node already belongs to a single class
            if len(np.unique(current_node.data)) == 1:
                current_node.terminal = True
                continue  # Move on to the next node in the queue

            # Split the data at this node based on the chosen feature
            current_node.split()

            current_node.calc_feature_importance(len(self.data))

            # Add the child nodes (created by the split) to the queue for further processing
            if current_node.children is not None:
                for child in current_node.children:
                    level_queue.put(child)

        ###########################################################################
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def predict(self, instance):
        """
        Predict a given instance
     
        Input:
        - instance: an row vector from the dataset. Note that the last element 
                    of this vector is the label of the instance.
     
        Output: the prediction of the instance.
        """
        pred = None
        ###########################################################################                                        
         # Start from the root node
        node = self.root  
        child_exists = True
        while (not node.terminal and child_exists):
            child_exists = False
            children_values_dict = dict(zip(node.children, node.children_values))
            for child, value in children_values_dict.items():
                if value == instance[node.feature]:
                    child_exists = True
                    # Advance to the next node for prediction
                    node = child
                    break  
        ###########################################################################
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return node.pred

    def calc_accuracy(self, dataset):
        """
        Predict a given dataset 
     
        Input:
        - dataset: the dataset on which the accuracy is evaluated
     
        Output: the accuracy of the decision tree on the given dataset (%).
        """
        accuracy = 0
        ###########################################################################
        good_predictions = 0
        total_predictions = len(dataset)
        
        for instance in dataset: #run through all instances and calculate accuracy
            prediction = self.predict(instance)
            if prediction == instance[len(instance)-1]:
                good_predictions += 1
        
        accuracy = (float(good_predictions) / total_predictions) 

        ###########################################################################
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return accuracy
        
    def depth(self):
        return self.root.depth()

def depth_pruning(X_train, X_validation):
    """
    Calculate the training and validation accuracies for different depths
    using the best impurity function and the gain_ratio flag you got
    previously. On a single plot, draw the training and testing accuracy 
    as a function of the max_depth. 

    Input:
    - X_train: the training data where the last column holds the labels
    - X_validation: the validation data where the last column holds the labels
 
    Output: the training and validation accuracies per max depth
    """
    training = []
    validation  = []
    root = None
    for max_depth in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        ###########################################################################
        root = DecisionTree(X_train, calc_entropy, gain_ratio=True, max_depth=max_depth)
        root.build_tree()
        training_accuracy = root.calc_accuracy(X_train)
        testing_accuracy = root.calc_accuracy(X_validation)
        training.append(training_accuracy)
        validation.append(testing_accuracy)
        ###########################################################################
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
    return training, validation


def chi_pruning(X_train, X_test):

    """
    Calculate the training and validation accuracies for different chi values
    using the best impurity function and the gain_ratio flag you got
    previously. 

    Input:
    - X_train: the training data where the last column holds the labels
    - X_validation: the validation data where the last column holds the labels
 
    Output:
    - chi_training_acc: the training accuracy per chi value
    - chi_validation_acc: the validation accuracy per chi value
    - depth: the tree depth for each chi value
    """
    chi_training_acc = []
    chi_testing_acc  = []
    depth = []

    ###########################################################################
    p_values=[1, 0.5, 0.25, 0.1, 0.05, 0.0001]
    for p in p_values:
        tree = DecisionTree(X_train, calc_entropy, chi=p, gain_ratio=True)
        tree.build_tree()
        tree_depth = calculate_tree_depth(tree.root)
        depth.append(tree_depth)        
        training_accuracy = tree.calc_accuracy(X_train)
        testing_accuracy = tree.calc_accuracy(X_test)
        chi_training_acc.append(training_accuracy)
        chi_testing_acc.append(testing_accuracy)
    ###########################################################################
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
        
    return chi_training_acc, chi_testing_acc, depth


def calculate_tree_depth(node):
    """
    Calculate the depth of the tree by adding each depth of a node to a list and returns the maximum value
    of the depth that we got on the list. By doing this we basically get a node and then go down to the leaf to get the
    depth of the whole tree.

    Input:
    - node: a node from a tree.

    Output: the depth of the tree.
    """
    # Check if we got to a leaf.
    if node.terminal:
        return node.depth

    # Check if the node has any children
    if not node.children:
        return node.depth
    
    # Creating a list that will contain the depth of the node's children.
    depths = []

    # Going over all the children of the node.
    for child in node.children:
        depths.append(calculate_tree_depth(child))
    return max(depths)


def count_nodes(node):
    """
    Count the number of node in a given tree
 
    Input:
    - node: a node in the decision tree.
 
    Output: the number of node in the tree.
    """
    ###########################################################################
    n_nodes = count_nodes_recursive(node,0)
    ###########################################################################
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return n_nodes

def count_nodes_recursive(current_node, sum):
    """
    Recursively counts the number of nodes in the tree starting from the current node.

    Inputs:
    - current_node: The current node being examined.
    - sum: The current count of nodes.

    Returns:
    - The total count of nodes in the subtree rooted at the current node.
    """
    if current_node==None:
        return 0

    # If the current node is a leaf, return 1
    if current_node.terminal:
        return 1

    # Initialize the count of child nodes
    child_count = 0
    
    # Traverse each child node
    for child in current_node.children:
        # Recursively count nodes in the subtree rooted at the child node
        child_count += count_nodes_recursive(child, sum)
    
    # Return the total count of nodes in the subtree rooted at the current node
    return 1 + child_count

    





