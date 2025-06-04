import pandas as pd
import argparse

from dataset import Dataset

TOY_DATASET = True

"""
This is a python script for analysing embeddings, e.g., from an XLS-R model fine-tuned for
language identification. Your dataset should be in 'data' folder and be tab delimited.
The dataset should at least contain a column 'language', and 'iso' with an iso639 code for each language, and
the embedding dimensions should be as columns D1 onwards. If you want to use 'accuracy...',
make sure you have 'pred_label' column with iso codes that are in the same format as 'iso' column.
NOTE:
Current implementation does not yet generalize, main function is to reproduce the results of a submitted paper.

"""


# Parse command line arguments
parser = argparse.ArgumentParser(description='This script visualizes embeddings. You can choose which classes to use for the LDA, how many components to calculate, what to visualize, and the method for dimensionality reduction. The default is to calculate the maximum number of components for the LDA, and to run all available visualizations. The default method for dimensionality reduction is LDA, and the default distance metric for the embeddings is cosine.',  epilog='The data should be in a tab delimited text file with at least iso codes and language names, followed by columns named D1 onwards.')
parser.add_argument('-f', '--file_path', type=str, help='Path to the pickle file. Default: data/embeddings.pkl', default='data/embeddings.pkl')
parser.add_argument('-c', '--transformation_class', type=str, help='Classes for supervised dimensionality reduction (only applicable for LDA). Default: language.', default='language')
parser.add_argument('-n', '--no_components', type=str, help='Number of dimensions after calculating dimensionality reduction, default: maximum for LDA (n_classes-1)', default='max')

parser.add_argument('-m', '--method', type=str, help='Method for dimensionality reduction, pca or lda, default: lda', default='lda')
parser.add_argument('-d', '--distance_metric', type=str, help='Metric for distance calculation. Default: cosine.', default='cosine')
parser.add_argument('-v', '--analysis_class', type=str, help='Class used for analysis. Default: language.', default='language')
parser.add_argument('-ag', '--agglomethod', type=str, help='Agglomerative clustering method. Default: average.', default='average')
parser.add_argument('-e', '--len_embeddings', type=str, help='Dimensionality of the input embeddings. Default: 512.', default=512)
parser.add_argument('-s', '--sample_size', type=int, help='Minimum sample size for analysis. Default: 20.', default=20)
parser.add_argument('-r', '--random_seed', type=int, help='Random seed for sampling. Default: None.', default=None)

# Convert arguments to dictionary
args = vars(parser.parse_args())

#
# Read dataset
if TOY_DATASET:
    dataframe = pd.read_pickle('data/sample_dataset.pkl')
    print("Using heavily reduced dataset for functionality testing.")
else:
    dataframe = pd.read_pickle(args['file_path'])


analyses = [
    "standard", # standard analyses, reproduces the statistical analyses and most visualizations using the full dataset 
    "consensus", # sample multiple subsets of the dataset and visualize a consensus tree
    "family" , # plots separate dendrograms for each language family
]
if TOY_DATASET:
    analyses = ["standard"] # only run standard analyses on the toy dataset

for analysis in analyses:
    
    print(f"\n***** performing {analysis} analyses *******")
  
    # create a data object with the dataframe and arguments
    data = (Dataset(dataframe, args, analysis_method=analysis)
        .prepare_languages() # normalize language names
        .sample_dataset() # sample dataset
        .scale_data() # scale data by subtracting mean
        .apply_dimensionality_reduction() # apply dimensionality reduction
        .get_means() # get centroids of your analysis class (e.g. language)
        .visualize()) # visualize a plot depending on the analysis type


