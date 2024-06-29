# Description: This file contains the functions to preprocess the skills and technologies of the candidates.

# Function to preprocess the skills
def preprocess_skills(skills):
    """
    Parameters: skills: list
    Returns: preprocessed_skills: list of lists
    """
    preprocessed_skills = []
    for skill in skills:
        preprocessed_skills.append(list(skill.split("/")))
    return preprocessed_skills

# Function to preprocess the technologies
def preprocess_technologies(technologies):
    """
    Parameters: technologies: list
    Returns: preprocessed_tech: list of lists
    """
    preprocessed_tech = []
    for tech in technologies:
        preprocessed_tech.append(list(tech.split("/")))
    return preprocessed_tech
