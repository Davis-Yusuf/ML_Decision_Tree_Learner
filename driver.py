"""
Machine learning
decision trees
"""
import time


from ml_lib.ml_util import DataSet
from decision_tree import DecisionTreeLearner

from ml_lib.crossval import cross_validation

from statistics import mean, stdev
    

def main():
    """
    Machine learning with decision trees.
    Runs cross validation on data sets and reports results/trees
    """
    # dataset = DataSet(name="mushrooms", target=0, attr_names=True)
    # dataset = DataSet(name="zoo", attr_names=True, exclude=[0])
    dataset = DataSet(name="restaurant", attr_names=True)
    # dataset = DataSet(name="grader.zoo", attr_names=True)
    tree = DecisionTreeLearner(dataset=dataset)
    # print(tree)
    print("---")
    tree.prune(0.05)
    # for i in range(12):
    #     print(tree.predict(dataset.examples[i]))


if __name__ == '__main__':
    main()
