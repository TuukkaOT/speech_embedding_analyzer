import pandas as pd
import argparse
from visualizer import Visualizer
from dataset import Dataset



"""
This is a python scrip for analysing embeddings, e.g., from an XLS-R model fine-tuned for
language identification. Your dataset should be in 'data' folder and be tab delimited.
The dataset should at least contain a column 'language', and 'iso' with an iso639 code for each language, and
the embedding dimensions should be as columns D1 onwards. If you want to use 'accuracy...',
make sure you have 'pred_label' column with iso codes that are in the same format as 'iso' column.
"""



# Parse command line arguments
parser = argparse.ArgumentParser(description='This script visualizes embeddings. You can choose which classes to use for the LDA, how many components to calculate, what to visualize, and the method for dimensionality reduction. The default is to calculate the maximum number of components for the LDA, and to run all available visualizations. The default method for dimensionality reduction is LDA, and the default distance metric for the embeddings is cosine.',  epilog='The data should be in a tab delimited text file with at least iso codes and language names, followed by columns named D1 onwards.')
parser.add_argument('-f', '--file_path', type=str, help='Path to the TSV file. Default: data/embeddings.tsv', default='data/embeddings.tsv')
parser.add_argument('-c', '--transformation_class', type=str, help='Classes for supervised dimensionality reduction (only applicable for LDA). Default: language.', default='language')
parser.add_argument('-n', '--no_components', type=str, help='Number of dimensions after calculating dimensionality reduction, default: maximum for LDA (n_classes-1)', default='max')
parser.add_argument('-a', '--analysis', type=str, help='What to analyze/visualize, (d)endrogram, (n)eighbornet, language (f)amily dendrograms, (c)onsensus tree, (s)scatter, statistical (a)nalyses, (all) visualizations. Default: all.', default='all')
parser.add_argument('-m', '--method', type=str, help='Method for dimensionality reduction, pca or lda, default: lda', default='lda')
parser.add_argument('-d', '--distance_metric', type=str, help='Metric for distance calculation. Default: cosine.', default='cosine')
parser.add_argument('-v', '--analysis_class', type=str, help='Class used for analysis. Default: language.', default='language')
parser.add_argument('-ag', '--agglomethod', type=str, help='Agglomerative clustering method. Default: average.', default='average')
parser.add_argument('-e', '--len_embeddings', type=str, help='Dimensionality of the input embeddings. Default: 512.', default=512)
parser.add_argument('-s', '--sample_size', type=int, help='Minimum sample size for analysis. Default: 20.', default=20)
parser.add_argument('-r', '--random_seed', type=int, help='Random seed for sampling. Default: None.', default=None)

# Convert arguments to dictionary
args = vars(parser.parse_args())

# dictionary for mapping analysis types to their descriptions
analysis_dict = {
    'd': 'a dendrogram',
    'n': 'a neighbornet',
    'f': 'language family dendrograms',
    'c': 'a consensus tree',
    's': 'a scatter plot',
    'a': 'statistical analyses'
}

# Read dataset
dataframe = pd.read_csv(args['file_path'], delimiter='\t')

# If the user wants to run all analyses, loop through the analysis types
if args['analysis'] == 'all':
    analyses = ['d', 'n', 'f', 'c', 's', 'a']
    for analysis in analyses:
        args['analysis'] = analysis
        print(f"Visualizing {analysis_dict[args['analysis']]}")

        # create a data object with the dataframe and arguments
        data = (Dataset(dataframe, args)
            .prepare_languages() # normalize language names
            .sample_dataset() # sample dataset
            .scale_data() # scale data by subtracting mean
            .apply_dimensionality_reduction() # apply dimensionality reduction
            .get_means() # get centroids of your analysis class (e.g. language)
            .visualize()) # visualize a plot depending on the analysis type


# else if the user wants to run a specific analysis, run that analysis
else:
    print(f"Visualizing {analysis_dict[args['analysis']]}")

    # create a data object with the dataframe and arguments
    data = (Dataset(dataframe, args)
        .prepare_languages() # normalize language names
        .sample_dataset() # sample dataset
        .scale_data() # scale data by subtracting mean
        .apply_dimensionality_reduction() # apply dimensionality reduction
        .get_means() # get centroids of your analysis class (e.g. language)
        .visualize()) # visualize a plot depending on the analysis type


# Check if the analysis type is valid
if args['analysis'] not in ['d', 'n', 'f', 'c', 's', 'a', 'all']:
    raise ValueError(f"Invalid analysis type: {args['analysis']}")