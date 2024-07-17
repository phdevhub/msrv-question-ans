from flask import Flask, jsonify
import json
from flask import request
from haystack.document_stores import InMemoryDocumentStore
from haystack.utils import fetch_archive_from_http
from haystack.pipelines.standard_pipelines import TextIndexingPipeline
from haystack.nodes import BM25Retriever
from haystack.nodes import FARMReader
from haystack.pipelines import ExtractiveQAPipeline
from haystack.utils import print_answers
from pprint import pprint
import os

app = Flask(__name__)

pprint("[INFO] : Initializing model")
document_store = InMemoryDocumentStore(use_bm25=True)

doc_dir = "data/qa_context_dataset"

output_dir=doc_dir,

files_to_index = [doc_dir + "/" + f for f in os.listdir(doc_dir)]
indexing_pipeline = TextIndexingPipeline(document_store)
indexing_pipeline.run_batch(file_paths=files_to_index)

pprint("[INFO] : Loading pipeline components")
retriever = BM25Retriever(document_store=document_store)
reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True)
pipe = ExtractiveQAPipeline(reader, retriever)

pprint("[INFO] : Finish initializing model")

def predict(query):
    return pipe.run(
        query=query, params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 5}}
    )

@app.route("/makeQuery", methods=["POST"])
def echo():
    return jsonify(predict(json.loads(request.get_data())["query"]))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
    