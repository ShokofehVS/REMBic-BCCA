import numpy as np
import random as rd
import preprocessing
import argparse
from BCCA import BiCorrelationClusteringAlgorithm


def logarithmic_transformation(data_test):
    # Do Logaritmic transformations on dataset that are suggested in the paper
    nonzero_mask = data_test != 0
    data_test_log_norm = np.zeros_like(data_test)
    data_test_log_norm[nonzero_mask] = np.log10(data_test[nonzero_mask] * (10 ** 5)) * 100

    return data_test_log_norm

def get_sample(data_test, y_cat, sample_size=None):

    if not sample_size:
        y_cat = y_cat.reset_index()
        return data_test, y_cat
    else:
        # Define sample of data to use CCA on smaller dataset.
        num_rows = len(data_test)
        random_indices = rd.sample(range(num_rows), sample_size)
        sample_data = data_test[random_indices]
        sample_labels = y_cat.iloc[random_indices]
        sample_labels = sample_labels.reset_index()
        print(sample_labels)
        return sample_data, sample_labels


def get_distribution_of_sample(sample_labels):

    labels = {"activating": [], "repressing": []}

    for label in labels:
        count = sample_labels["Predicted_function"].value_counts().get(label, 0)
        labels[label] = count


def run_BCCA(data):

    # missing value imputation suggested by Cheng and Church
    missing = np.where(data < 0.0)
    data[missing] = np.random.randint(low=0, high=800, size=len(missing[0]))

    print("nomal BCCA:")

    bcca = BiCorrelationClusteringAlgorithm(correlation_threshold=0.9, min_cols=3)
    biclusters = bcca.run(data)
    print(biclusters)

    return biclusters



def main(sample_size=None):

    data = preprocessing.preprocessing()
    test_data = data[0][0]
    y_cat_test = data[0][1]
    all_cat_test = data[0][2]

    test_data = logarithmic_transformation(test_data)
    sample_data, sample_labels = get_sample(test_data, y_cat_test, sample_size=sample_size)
    get_distribution_of_sample(sample_labels)
    biclustering_test = run_BCCA(test_data)
    # results = format_cca_results(biclustering_test, all_cat_test, sample_labels)

    # mult_acc = calc_multi_classification(results)
    # bin_acc = calculate_binary_classification(results)


if __name__ == '__main__':
    # Using argparse to specify the sample_size parameter when running the code
    parser = argparse.ArgumentParser()
    parser.add_argument("sample_size", type=int, nargs='?', default=None, help="Sample size for CCA")
    args = parser.parse_args()

    main(sample_size=None)





