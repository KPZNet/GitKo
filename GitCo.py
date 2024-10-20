import requests
import os
from git import Repo

def get_github_repos(username):
    url = f"https://api.github.com/users/{username}/repos"
    response = requests.get(url)
    
    if response.status_code == 200:
        repos = response.json()
        return [repo['name'] for repo in repos]
    else:
        return f"Error: Unable to fetch repositories for user {username}"


def clone_github_repos(username, local_repos_path):
    repos = get_github_repos(username)
    if isinstance(repos, list):
        for repo_name in repos:
            repo_url = f"https://github.com/{username}/{repo_name}.git"
            repo_path = os.path.join(local_repos_path, repo_name)
            if not os.path.exists(repo_path):
                try:
                    Repo.clone_from(repo_url, repo_path)
                    print(f"Cloned {repo_name} into {repo_path}")
                except Exception as e:
                    print(f"Error cloning repo {repo_name}: {e}")
            else:
                print(f"Repository {repo_name} already exists at {repo_path}")
    else:
        print(repos)

def pull_latest_for_repo(local_repo_path):
    if os.path.isdir(local_repo_path):
        try:
            repo = Repo(local_repo_path)
            origin = repo.remotes.origin
            origin.pull()
            print(f"Pulled latest changes for repo at {local_repo_path}")
        except Exception as e:
            print(f"Error pulling latest changes for repo at {local_repo_path}: {e}")
    else:
        print(f"{local_repo_path} is not a directory")



def fetch_origin_for_all_repos(local_repos_path):
    for repo_name in os.listdir(local_repos_path):
        repo_path = os.path.join(local_repos_path, repo_name)
        if os.path.isdir(repo_path):
            try:
                repo = Repo(repo_path)
                origin = repo.remotes.origin
                origin.fetch()
                print(f"Fetched origin for repo {repo_name}")
            except Exception as e:
                print(f"Error fetching origin for repo {repo_name}: {e}")
        else:
            print(f"{repo_path} is not a directory")



def update_all_repos_skip_dirty(local_repos_path):
    for repo_name in os.listdir(local_repos_path):
        repo_path = os.path.join(local_repos_path, repo_name)
        if os.path.isdir(repo_path):
            try:
                repo = Repo(repo_path)
                if repo.is_dirty():
                    print(f"Skipping dirty repo {repo_name}")
                    continue
                active_branch = repo.active_branch.name
                for remote in repo.remotes:
                    remote.fetch()
                    for ref in remote.refs:
                        if ref.remote_head not in repo.branches:
                            repo.create_head(ref.remote_head, ref)
                        repo.git.checkout(ref.remote_head)
                        repo.git.pull(remote.name, ref.remote_head)
                        print(f"Updated branch {ref.remote_head} from remote {remote.name} in repo {repo_name}")
                repo.git.checkout(active_branch)
            except Exception as e:
                print(f"Error updating branches in repo {repo_name}: {e}")
        else:
            print(f"{repo_path} is not a directory")


def list_all_branches_and_active_branch_for_all_repos(local_repos_path):
    for repo_name in os.listdir(local_repos_path):
        repo_path = os.path.join(local_repos_path, repo_name)
        if os.path.isdir(repo_path):
            try:
                repo = Repo(repo_path)
                branches = [branch.name for branch in repo.branches]
                active_branch = repo.active_branch.name
                print(f"Branches for repo {repo_name}: {branches}")
                print(f"Active branch for repo {repo_name}: {active_branch}")
            except Exception as e:
                print(f"Error listing branches for repo {repo_name}: {e}")
        else:
            print(f"{repo_path} is not a directory")

# Example usage
if __name__ == "__main__":
    username = "kpznet"  # Replace with the GitHub username you want to query
    #clone_github_repos(username, "C:/Users/kence/Documents/KProjects/gitprojects")  # Clone all repos from the user into the "repos" directory
    update_all_repos_skip_dirty("C:/Users/kence/Documents/KProjects/gitprojects")  # Update all repos in the "repos" directory
    list_all_branches_and_active_branch_for_all_repos("C:/Users/kence/Documents/KProjects/gitprojects")  # List all branches for all repos in the "repos" directory
