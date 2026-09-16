from job_search_agent.scoring import calculate_overall_score


def test_calculate_overall_score():
    weights = {
        "skills": 0.35,
        "education": 0.20,
        "experience": 0.25,
        "career_relevance": 0.20,
    }

    score = calculate_overall_score(
        skills_score=90,
        education_score=100,
        experience_score=70,
        career_relevance_score=95,
        weights=weights,
    )

    assert score == 88.0