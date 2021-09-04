# Hands On Git

## What is git

## Why git

## Hands On commands

### Interation with github

#### Clone

clone a githut repo.

`git clone <github-repo>`

#### Initialization

initialize your local repo.

`git init`

#### Push

push your local repo to remote.

`git push`

#### Pull

pull remote repo to local.

`git pull`

### Staging

#### Check status

check out current status, e.g. modified files are not been added, added files are not been commited.

`git status`

#### staging file

`git add <file name>`

### Commit code

**Highly Recommend:** 

`git commit`

write comprehensive commit message is of great importance for you project development.

**Don’t recommend:**

`git commit -m “message..”`

#### stage and git: 

if untracked files is too many, stage and commit each part of them once a time.

**Don’t recommend:**

`git commit -am “message”` 

### RollBack

#### rollback staging

`git checkout <added file>`

#### reset commit

`git reset HEAD^1` 

rollback to last commit

### History Log

`git log`

### Checkout commit/ branch

`git checkout <commit/branch>`

### Ignoring files

`.gitignore` list files or folders you do not want to track

### branch

#### create new branch

`git branch <branch-name>`

#### change branch

`git checkout <branch-name>`

Or

`git checkout -b branchname`

#### merging branch to main

`git checkout main/master`

`git merge <branch-name>`

#### delete branch 

when a features is well developed, then merge it to main and delete it.

`git branch -d <branch-name>`