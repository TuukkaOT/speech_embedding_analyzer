# speech_embedding_analyzer

A tool for analysing embeddings, e.g., from an XLS-R model fine-tuned for
language identification. Your dataset should be in 'data' folder and be tab delimited.
The dataset should at least contain a column 'language', 'language_family', and 'iso' with an iso639-3 code for each language, and
the embedding dimensions should be as columns D1 onwards.

Install the required pip packages in your environment:
pip install -r requirements.txt

To create all available visualizations, run the "main.py" script without arguments. The embedding file needs to be in data folder as "embeddings.pkl".

If your embeddings file does not contain columns 'latitude' and 'longitude', the script will try to match the iso codes with coordinates from Glottolog. If there are iso codes not corresponding to coordinates in Glottolog, the script will raise an error.

The Nexus and Newick tree files have been tested to work on Splitstree 6.4.13.