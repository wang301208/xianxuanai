"""Scaffolding for distributed skill search using Spark or MapReduce."""
from __future__ import annotations

from typing import List


def spark_search(
    vector_index_path: str, query_embedding: List[float], n_results: int = 3
):
    """Example scaffold for running a distributed search on a Spark cluster.

    Parameters
    ----------
    vector_index_path: str
        Path to a persisted vector index dataset accessible to the cluster.
    query_embedding: List[float]
        Embedding to query with.
    n_results: int
        Number of results to return.

    Notes
    -----
    This function is a placeholder demonstrating how Spark could be used to
    perform the similarity search in parallel across a cluster. A full
    implementation would load the vector index as a DataFrame and compute
    cosine similarities using Spark SQL or UDFs.
    """
    try:
        from pyspark.sql import SparkSession
    except Exception as e:  # pragma: no cover - pyspark optional
        raise RuntimeError("pyspark is required for spark_search") from e

    spark = SparkSession.builder.appName("SkillSearch").getOrCreate()
    try:
        from pyspark.sql import functions as F
        from pyspark.sql.types import FloatType
        import math

        # Broadcast query embedding to workers
        query_broadcast = spark.sparkContext.broadcast(query_embedding)

        # Load vector index; expect JSON lines with `id` and `embedding`
        df = spark.read.json(vector_index_path)

        # Define UDF for cosine similarity
        def cosine(vec):
            q = query_broadcast.value
            dot = sum(a * b for a, b in zip(vec, q))
            norm_vec = math.sqrt(sum(a * a for a in vec))
            norm_q = math.sqrt(sum(a * a for a in q))
            return float(dot / (norm_vec * norm_q)) if norm_vec and norm_q else 0.0

        cosine_udf = F.udf(cosine, FloatType())

        results_df = (
            df.withColumn("similarity", cosine_udf(F.col("embedding")))
            .orderBy(F.col("similarity").desc())
            .limit(n_results)
        )

        rows = results_df.select("id", "similarity").collect()
        return [(row["id"], row["similarity"]) for row in rows]
    finally:
        spark.stop()


def map_reduce_search(data_path: str, query_embedding: List[float], n_results: int = 3):
    """Placeholder for a MapReduce-style distributed search.

    This function outlines how a map and reduce phase could be structured to
    distribute the workload across a cluster. The map phase would compute
    similarities for partitions of the dataset, and the reduce phase would
    aggregate the top results.
    """
    try:
        from mrjob.job import MRJob
    except Exception as e:  # pragma: no cover - mrjob optional
        raise RuntimeError("mrjob is required for map_reduce_search") from e

    query = query_embedding

    class MRSimilaritySearch(MRJob):
        """MRJob computing cosine similarity for each record."""

        def mapper(self, _, line):  # pragma: no cover - executed in MRJob
            import json
            import math
            data = json.loads(line)
            vec = data["embedding"]
            dot = sum(a * b for a, b in zip(vec, query))
            norm_vec = math.sqrt(sum(a * a for a in vec))
            norm_q = math.sqrt(sum(a * a for a in query))
            sim = float(dot / (norm_vec * norm_q)) if norm_vec and norm_q else 0.0
            yield None, (sim, data["id"])

        def reducer(self, _, values):  # pragma: no cover - executed in MRJob
            import heapq
            top = heapq.nlargest(n_results, values)
            for sim, idx in top:
                yield idx, sim

    mr_job = MRSimilaritySearch(args=[data_path, "--no-conf", "--runner=inline"])
    with mr_job.make_runner() as runner:
        runner.run()
        output = list(mr_job.parse_output(runner.cat_output()))

    output.sort(key=lambda x: x[1], reverse=True)
    return output[:n_results]
