"""
demo.py – Run the pipeline end-to-end with a synthetic corpus.

    cd gcp_graphrag/
    python demo.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from pipeline import GCPGraphRAGPipeline
from config.settings import PipelineConfig, RetrievalConfig, GraphConfig, GCPConfig
from utils.evaluation import evaluate_pipeline

CORPUS = [
    {"content": "The DGCA issued drone regulations in 2021. Commercial drones require a UAOP.",
     "modality": "text", "source": "dgca_gov_in", "lang": "en", "provenance_score": 0.95},
    {"content": "In 2023, India updated drone regulations to include remote ID for drones >250g.",
     "modality": "text", "source": "meity_2023",  "lang": "en", "provenance_score": 0.92},
    {"content": "भारत में ड्रोन उड़ाने के लिए डिजिटल स्काई प्लेटफॉर्म पर पंजीकरण अनिवार्य है।",
     "modality": "text", "source": "hindi_gazette","lang": "hi", "provenance_score": 0.88},
    {"content": "No-fly zones: within 5 km of airports, 3 km of defence establishments.",
     "modality": "text", "source": "dgca_gov_in", "lang": "en", "provenance_score": 0.95},
    {"content": "Drone categories: micro ≤250g, small 250g–2kg, medium 2–25kg, large >25kg.",
     "modality": "text", "source": "pib_gov_in",  "lang": "en", "provenance_score": 0.90},
    {"content": "Remote pilot certificate mandatory for drones over 2 kg.",
     "modality": "text", "source": "aviation_ministry","lang":"en","provenance_score": 0.91},
    {"content": "2023 amendment adds geo-fencing and real-time tracking for commercial operators.",
     "modality": "text", "source": "meity_2023",  "lang": "en", "provenance_score": 0.89},
    {"content": "Agricultural drones may operate in Green Zones without prior permission.",
     "modality": "text", "source": "agriculture_ministry","lang":"en","provenance_score":0.87},
]

config = PipelineConfig(
    retrieval=RetrievalConfig(top_k=8),
    graph=GraphConfig(edge_threshold=0.4, num_gat_layers=2),
    gcp=GCPConfig(relevance_weight=0.5, coherence_weight=0.3, provenance_weight=0.2),
)
pipeline = GCPGraphRAGPipeline(config=config)

print("\n" + "="*60)
print("  GCP-GraphRAG Demo")
print("="*60)
print(f"\n[1/4] Indexing {len(CORPUS)} documents…")
pipeline.index_corpus(CORPUS)

QUERY = "What are the drone regulations in India in 2023?"
print(f"\n[2/4] Query: '{QUERY}'\n")
result = pipeline.run(query=QUERY, lang="en")

print("-"*60)
print("PRUNED EVIDENCE")
print("-"*60)
for ev in result["pruned_evidence"]:
    print(f"  [{ev['modality'].upper():<5} | imp={ev['importance_score']:.3f} | "
          f"src={ev['source']}]\n  {str(ev['content'])[:120]}\n")

print("-"*60)
print("GRAPH STATS")
for k, v in result["graph_stats"].items():
    print(f"  {k:<18}: {v:.4f}" if isinstance(v, float) else f"  {k:<18}: {v}")

print(f"\n  Latency: {result['latency_ms']:.1f} ms")

print("\n" + "-"*60)
print("EVALUATION")
GT = ("In 2023, India requires drone registration on the Digital Sky platform. "
      "Remote ID mandatory for drones >250g. Commercial drones need UAOP.")
metrics = evaluate_pipeline(
    prediction=GT, ground_truth=GT,
    retrieved_ids=[e["id"] for e in result["pruned_evidence"]],
    relevant_ids={0, 1, 3, 4, 6},
    evidence=result["pruned_evidence"],
    token_stats=result["token_stats"], k=5,
)
for metric, score in metrics.items():
    print(f"  {metric:<30}: {score:.4f}")

print("\n" + "="*60 + "\n")
