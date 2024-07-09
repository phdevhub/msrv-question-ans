from haystack.document_stores import InMemoryDocumentStore
from haystack.utils import fetch_archive_from_http
from haystack.pipelines.standard_pipelines import TextIndexingPipeline
from haystack.nodes import BM25Retriever
from haystack.nodes import FARMReader
from haystack.pipelines import ExtractiveQAPipeline
from haystack.utils import print_answers
from pprint import pprint
import os

document_store = InMemoryDocumentStore(use_bm25=True)

doc_dir = "data/qa_context_dataset"

output_dir=doc_dir,

files_to_index = [doc_dir + "/" + f for f in os.listdir(doc_dir)]
indexing_pipeline = TextIndexingPipeline(document_store)
indexing_pipeline.run_batch(file_paths=files_to_index)

retriever = BM25Retriever(document_store=document_store)

reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True)

pipe = ExtractiveQAPipeline(reader, retriever)

commands = ["/q", "/init"]
query = "/init"
while query != "/q":
    query = input("? ")
    if query not in commands:
        prediction = pipe.run(
            query=query, params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 5}}
        )

        pprint(prediction)

        print_answers(prediction, details="minimum")  ## Choose from `minimum`, `medium`, and `all`
