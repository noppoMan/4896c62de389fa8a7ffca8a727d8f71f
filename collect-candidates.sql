/**
his SQL extracts candidates for REV and NON_REV from the data of ghs(https://github.com/seart-group/ghs?tab=readme-ov-file).
Obtain the dump data from the following URL and import it into MySQL for use.
https://www.dropbox.com/scl/fo/lqvp1mhsg0ezp2sgs0xdk/h/20240801?dl=0&subfolder_nav_tracking=1
**/

/**
view for rev_class_candidates
DATEDIFF('2024-08-06', r.last_commit) > 180 is a query condition used to broadly identify projects, other than those marked as archived, whose activity levels have declined.
The determination of dormancy described in the paper is made by cloning the target repository and analyzing the commit logs.
**/
CREATE OR REPLACE VIEW rev_class_candidates AS
SELECT 
    r.id,
    r.name,
    r.stargazers,
    r.commits,
    r.contributors,
    r.language_id,
    r.created_at,
    r.last_commit,
    r.archived,
    l.name as language_name,
    DATEDIFF('2024-08-06', r.last_commit) as days_inactive,
    'REV_CANDIDATE' as project_class
FROM git_repo r
LEFT JOIN language l ON r.language_id = l.id
WHERE 
    r.stargazers >= 2538
    AND r.commits >= 500
    AND r.contributors >= 23
    AND (
        r.archived = 1 OR
        DATEDIFF('2024-08-06', r.last_commit) > 180
    );

-- view for non_rev_class_candidates
CREATE OR REPLACE VIEW non_rev_class_candidates AS
SELECT 
    r.id,
    r.name,
    r.stargazers,
    r.commits,
    r.contributors,
    r.language_id,
    r.created_at,
    r.last_commit,
    r.archived,
    l.name as language_name,
    DATEDIFF('2024-08-06', r.last_commit) as days_inactive,
    DATEDIFF(r.last_commit, r.created_at) as delta,
    'NON_REV_CANDIDATE' as project_class
FROM git_repo r
LEFT JOIN language l ON r.language_id = l.id
WHERE 
    r.stargazers >= 2538
    AND r.commits >= 500
    AND r.contributors >= 23
    AND r.archived = 0
    AND DATEDIFF('2024-08-06', r.last_commit) <= 365;