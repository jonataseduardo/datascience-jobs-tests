from pathlib import Path

import lxml.etree as ET
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from bertopic.vectorizers import ClassTfidfTransformer
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP


def get_abstract_from_xml(xml_path: str) -> str:
    try:
        tree = ET.parse(xml_path)
        abstract_html = str(tree.find(".//AbstractNarration").text)
        abstract = abstract_html.replace("&lt;br/&gt;", "")
        return abstract
    except Exception as e:
        print("Error: ", e)


files = sorted(Path("data").rglob("*.xml"))
abstracts = [_ for _ in [get_abstract_from_xml(str(f)) for f in files] if _ != "None"]

# Step 1 - Extract embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Step 2 - Reduce dimensionality
umap_model = UMAP(n_neighbors=30, n_components=5, min_dist=0.0, metric="cosine")

# Step 3 - Cluster reduced embeddings
hdbscan_model = HDBSCAN(
    min_cluster_size=50,
    metric="euclidean",
    cluster_selection_method="eom",
    prediction_data=True,
)

# Step 4 - Tokenize topics
vectorizer_model = CountVectorizer(stop_words="english")

# Step 5 - Create topic representation
ctfidf_model = ClassTfidfTransformer()

# Step 6 -  Fine-tune topic representations with
# `bertopic.representation` model
representation_model = KeyBERTInspired()

# All steps together
topic_model = BERTopic(
    embedding_model=embedding_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer_model,
    ctfidf_model=ctfidf_model,
    representation_model=representation_model,
    top_n_words=10,
)

topics, probs = topic_model.fit_transform(abstracts)

topic_model.get_topic_info().iloc[:, :3]

# Reduce outliers
# new_topics = topic_model.reduce_outliers(abstracts, topics)
# Reduce outliers with pre-calculate embeddings instead
new_topics = topic_model.reduce_outliers(
    abstracts, topics, strategy="embeddings", embeddings=embeddings
)

topic_model.update_topics(abstracts, topics=new_topics)
topic_model.get_topic_info().Name

topic_model.visualize_topics()
topic_model.visualize_hierarchy()

embeddings = embedding_model.encode(abstracts, show_progress_bar=True)
reduced_embeddings = umap_model.fit_transform(embeddings)

topic_model.visualize_documents(
    abstracts, reduced_embeddings=reduced_embeddings, hide_annotations=True
)
