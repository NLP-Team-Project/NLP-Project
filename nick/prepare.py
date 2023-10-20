import os
import json
from typing import Dict, List, Optional, Union, cast
import requests

from env import github_token, github_username

# TODO: Make a github personal access token.
#     1. Go here and generate a personal access token https://github.com/settings/tokens
#        You do _not_ need select any scopes, i.e. leave all the checkboxes unchecked
#     2. Save it in your env.py file under the variable `github_token`
# TODO: Add your github username to your env.py file under the variable `github_username`
# TODO: Add more repositories to the `REPOS` list below.

headers = {"Authorization": f"token {github_token}",
           "User-Agent": github_username}

if headers["Authorization"] == "token " or headers["User-Agent"] == "":
    raise Exception("You need to follow the instructions marked TODO in this script before trying to use it")


# REPOS = []


def github_api_request(url: str) -> Union[List, Dict]:
    response = requests.get(url, headers=headers)
    response_data = response.json()
    if response.status_code != 200:
        raise Exception(f"Error response from github api! status code: {response.status_code}, "
                        f"response: {json.dumps(response_data)}")
    return response_data


def get_repo_language(repo: str) -> str:
    url = f"https://api.github.com/repos/{repo}"
    repo_info = github_api_request(url)
    if type(repo_info) is dict:
        repo_info = cast(Dict, repo_info)
        return repo_info.get("language", None)
    raise Exception(f"Expecting a dictionary response from {url}, instead got {json.dumps(repo_info)}")


def get_repo_contents(repo: str) -> List[Dict[str, str]]:
    url = f"https://api.github.com/repos/{repo}/contents/"
    contents = github_api_request(url)
    if type(contents) is list:
        contents = cast(List, contents)
        return contents
    raise Exception(
        f"Expecting a list response from {url}, instead got {json.dumps(contents)}")


def get_readme_download_url(files: List[Dict[str, str]]) -> str:
    """
    Takes in a response from the github api that lists the files in a repo and
    returns the url that can be used to download the repo's README file.
    """
    for file in files:
        if file["name"].lower().startswith("readme"):
            return file["download_url"]
    return ""


def process_repo(repo: str) -> Dict[str, str]:
    """
    Takes a repo name like "gocodeup/codeup-setup-script" and returns a
    dictionary with the language of the repo and the readme contents.
    """
    print(repo)
    contents = get_repo_contents(repo)
    try:
        readme_contents = requests.get(get_readme_download_url(contents)).text
    except:
        readme_contents = 'failedreadme'
    return {
        "repo": repo,
        "language": get_repo_language(repo),
        "readme_contents": readme_contents,
    }


def scrape_github_data(REPOS) -> List[Dict[str, str]]:
    """
    Loop through all the repos and process them. Returns the processed data.
    """
    return [process_repo(repo) for repo in REPOS]


if __name__ == "__main__":
    data = scrape_github_data()
    json.dump(data, open("data2.json", "w"), indent=1)


def search_github_repositories(search_query, repository_type="repositories", per_page=100):
    # Define the base URL for the GitHub API
    base_url = "https://api.github.com/search/repositories"
    # Set up headers with your GitHub token and user-agent
    headers = {
        "Authorization": f"token {github_token}",
        "User-Agent": github_username
    }
    # Initialize an empty list to store repositories
    all_repositories = []
    # Initialize variables for pagination
    page = 1
    total_repos = 0
    while total_repos < 1000:
        params = {"q": search_query,
                  "type": repository_type,
                  "per_page": per_page,
                  "page": page}
        # Send a GET request to the GitHub API
        response = requests.get(base_url, headers=headers, params=params)
        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            data = response.json()
            # Extract the repositories from the current page
            repositories = data.get("items", [])
            # Append the current page's repositories to the list
            all_repositories.extend(repositories)
            # Update the total count of repositories retrieved
            total_repos += len(repositories)
            # Increment the page number for the next page
            page += 1
        else:
            print(f"Failed to retrieve data. Status code: {response.status_code}")
            break
    return all_repositories