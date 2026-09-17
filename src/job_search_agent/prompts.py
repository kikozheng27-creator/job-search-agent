from job_search_agent.candidate import CandidateProfile


EXTRACTION_RULES = """
Extraction rules (facts only, do not infer beyond the text):

- company, job_title, location: copy from the posting. Use null for location
  if it is not stated.
- employment_type and seniority_level: use "unknown" unless the posting is
  explicit.
- minimum_years_experience: the lowest number of years the posting requires.
  For a range such as "3-5 years", use 3. Use null if no number is given.
- required_skills vs preferred_skills: put a skill in preferred_skills when
  the posting words it as "preferred", "nice to have", or "a plus".
- Normalize every skill to a concise skill, tool, or competency name, the way
  it would appear on a resume: "R", "SAS", "survival analysis", "mixed
  models", "CDISC ADaM". Do not copy prose fragments from the posting. Strip
  qualifiers such as "strong", "expert-level", "deep knowledge of",
  "experience with", "familiarity with", and "preferred". Split a bullet
  listing several skills into one entry per skill.
- sponsorship: classify work-authorization language into exactly one of:
    * "not_mentioned"  - the posting says nothing about work authorization.
      This is the default and the correct answer when in doubt.
    * "offered"        - the posting says sponsorship is available.
    * "not_offered"    - the employer declines to sponsor a visa but does not
      rule out candidates who are already authorized. Examples: "we are
      unable to provide visa sponsorship", "no H-1B sponsorship",
      "sponsorship is not available for this role".
    * "permanent_authorization_required" - the posting rules out temporary or
      student authorization. Examples: "no F-1 candidates", "no OPT or STEM
      OPT candidates", "permanent unrestricted work authorization required",
      "must not now or in the future require sponsorship".
    * "citizenship_required" - the posting requires US citizenship, permanent
      residency, or a security clearance.
  The difference between "not_offered" and "permanent_authorization_required"
  matters: the first only declines future sponsorship, while the second
  rejects a candidate whose authorization is temporary. Choose
  "permanent_authorization_required" only when the posting clearly excludes
  temporary or student authorization.
  Any answer other than "not_mentioned" requires a sentence you can quote
  verbatim in sponsorship_language. If no sentence in the posting mentions
  work authorization, sponsorship, visas, or citizenship, then the answer is
  "not_mentioned" and sponsorship_language is null. Never infer a stance from
  the absence of such a sentence.
- sponsorship_language: the exact sentence from the posting, or null.
- salary_range: only if the posting states compensation. Otherwise null.
"""

SCORING_RULES = """
Score each component from 0 to 100, judging the candidate against this posting:

- skills: overlap between the candidate's skills and the required skills,
  weighting required above preferred.
- education: how well the candidate's degrees and fields match the required
  degree.
- experience: compare the candidate's years and level against what the
  posting asks. A candidate who meets or exceeds the stated minimum scores at
  least 85; do not deduct for having less than the top of a range, so a
  candidate with 0.5 years meets a "0-2 years" requirement in full. Use the
  85-100 band to reflect how relevant and substantial that experience is, and
  reserve scores below 85 for candidates genuinely short of the minimum,
  scaled by how far short they fall. When the posting states no minimum,
  judge relevance alone rather than assuming a requirement.
- career_relevance: how well this role advances the candidate's target job
  families and preferred industries.

Also provide:
- strengths: concrete reasons the candidate fits.
- missing_requirements: specific requirements the candidate does not meet.
- reasoning: two or three sentences explaining the component scores.

Do not compute an overall score.
Do not make an apply or skip recommendation.
Both are calculated separately in code.
"""


def build_evaluation_prompt(
    profile: CandidateProfile,
    job_description: str,
) -> str:
    return f"""You are evaluating how well a candidate matches a job posting.

Candidate profile:
{profile.model_dump_json(indent=2)}

Job posting:
{job_description}
{EXTRACTION_RULES}
{SCORING_RULES}"""
