from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import chromadb

from app.config import Settings


@dataclass(slots=True)
class CollectionRecord:
    record_id: str
    document: str
    metadata: dict[str, Any]
    embedding: list[float]


class ChromaRepository:
    def __init__(self, settings: Settings):
        self.settings = settings

        self.client = self._build_client()

        self.students = self.client.get_or_create_collection(
            name=self.settings.portfolio_collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        self.jobs = self.client.get_or_create_collection(
            name=self.settings.job_collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.applications = self.client.get_or_create_collection(
            name="applications"
        )

    def _build_client(self):
        return chromadb.PersistentClient(
            path=str(self.settings.chroma_path)
        )

    def _upsert(self, collection, record: CollectionRecord):
        collection.upsert(
            ids=[record.record_id],
            documents=[record.document],
            metadatas=[record.metadata],
            embeddings=[record.embedding],
        )

    def upsert_student(
        self,
        record: CollectionRecord,
    ):
        self._upsert(
            self.students,
            record,
        )

    def upsert_job(
        self,
        record: CollectionRecord,
    ):
        self._upsert(
            self.jobs,
            record,
        )

    def query_students(
        self,
        query_embedding: list[float],
        n_results: int = 25,
    ):
        return self.students.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

    def query_jobs(
        self,
        query_embedding: list[float],
        n_results: int = 25,
    ):
        return self.jobs.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

    def fetch_all_students(self):
        return self.students.get(
            include=[
                "documents",
                "metadatas",
            ]
        )
    

    def fetch_all_jobs(self):
        return self.jobs.get(
            include=[
                "documents",
                "metadatas",
            ]
        )
    
    
    def upsert_application(
        self,
        application_id: str,
        metadata: dict,
    ):
     self.applications.upsert(
        ids=[application_id],
        documents=[
            f"{metadata['student_id']} applied to {metadata['job_id']}"
        ],
        metadatas=[metadata],
        embeddings=[[0.0] * 384]
    )

    def fetch_applications_by_job(
        self,
        job_id: str,
    ):
        all_apps = self.applications.get()

        result = []

        for meta in all_apps["metadatas"]:
            if meta["job_id"] == job_id:
                result.append(meta)

        return result
    
    def fetch_student_by_id(
        self,
        student_id: str,
    ):
        result = self.students.get(
        where={
            "student_id": student_id
        }
    )

        if not result["ids"]:
         return None

        return result["metadatas"][0]
    