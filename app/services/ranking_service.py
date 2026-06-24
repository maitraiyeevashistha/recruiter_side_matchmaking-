from app.services.store import ChromaRepository


class RankingService:

    def __init__(
        self,
        repo: ChromaRepository,
    ):
        self.repo = repo

    def rank_job_candidates(
        self,
        job_id: str,
    ):
        applications = (
            self.repo.fetch_applications_by_job(
                job_id
            )
        )

        results = []

        for app in applications:

            student = (
                self.repo.fetch_student_by_id(
                    app["student_id"]
                )
            )

            if not student:
                continue

            assignment_score = (
                app.get(
                    "assignment_score",
                    0
                )
            )

            final_score = (
                assignment_score / 100
            )

            results.append(
                {
                    "student_id": app["student_id"],
                    "name": student.get(
                        "name",
                        ""
                    ),
                    "assignment_score": assignment_score,
                    "final_score": final_score,
                }
            )

        results.sort(
            key=lambda x: x["final_score"],
            reverse=True,
        )

        return results