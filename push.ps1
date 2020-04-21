clear
#update environment with new packages
conda env export > gm.yml

# add
git add -A

# commit
$commitMessage = Read-Host -Prompt 'Write a commit message: '
git commit -m "$commitMessage"

#push
git push -u origin master