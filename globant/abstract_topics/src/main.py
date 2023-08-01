from pathlib import Path
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP

import lxml.etree as ET

from sklearn.datasets import fetch_20newsgroups

docs = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))["data"]


def get_abstract_from_xml(xml_path: str) -> str:
    try:
        tree = ET.parse(xml_path)
        abstract_html = str(tree.find(".//AbstractNarration").text)
        abstract = abstract_html.replace("&lt;br/&gt;", "")
        return abstract
    except Exception as e:
        print("Error: ", e)


files = sorted(Path("../data").rglob("*.xml"))

abstracts = [get_abstract_from_xml(str(f)) for f in files]

topic_model = BERTopic()

topics, probs = topic_model.fit_transform(abstracts)

topics

topic_model.get_topic(-1)

topic_model.visualize_topics(custom_labels=True)

topic_model.visualize_hierarchy(custom_labels=True)


# Pre-calculate embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedding_model.encode(abstracts, show_progress_bar=True)


reduced_embeddings = UMAP(
    n_neighbors=10, n_components=2, min_dist=0.0, metric="cosine"
).fit_transform(embeddings)


topic_model.visualize_documents(
    topics,
    reduced_embeddings=reduced_embeddings,
    custom_labels=True,
    hide_annotations=True,
)
