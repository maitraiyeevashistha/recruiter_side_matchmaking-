from app.services.store import ChromaRepository


class ApplicationService:

    def __init__(
        self,
        repo: ChromaRepository,
    ):
        self.repo = repo

    def apply(
        self,
        application_id,
        student_id,
        job_id,
        assignment_score,
    ):

        metadata = {
            "application_id": application_id,
            "student_id": student_id,
            "job_id": job_id,
            "assignment_score": assignment_score,
        }

        self.repo.upsert_application(
            application_id,
            metadata,
        )

        return application_id

    def get_job_applications(
        self,
        job_id,
    ):
        return self.repo.fetch_applications_by_job(
            job_id
        )