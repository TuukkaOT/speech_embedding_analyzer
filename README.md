# speech_embedding_analyzer

A tool for analysing embeddings, e.g., from an XLS-R model fine-tuned for
language identification. Your dataset should be in 'data' folder and be tab delimited.
The dataset should at least contain a column 'language', 'language_family', and 'iso' with an iso639-3 code for each language, and
the embedding dimensions should be as columns D1 onwards.

Install the required pip packages in your environment:
pip install -r requirements.txt

To create the visualizations and analyses, run the 'main.py' script without arguments.

If your embeddings file does not contain columns 'latitude' and 'longitude', the script will try to match the iso codes with coordinates from Glottolog. If there are iso codes not corresponding to coordinates in Glottolog, the script will raise an error.

The Nexus and Newick tree files have been tested to work on Splitstree 6.4.13.

'''diff
- NOTE:
At the moment the analyzer is only intended for reproducing the results of a submitted paper and thus runs only on the accompanying dataset.''' As of now a small sample dataset is available in the repository as 'embeddings_toy_dataset.pkl'. To run all the analyses on the full dataset, the embedding file should to be in data folder as "embeddings.pkl" and the TOY_DATASET variable to be set to 'False' in 'main.py'.
