#!/usr/bin/env bash
# Initialize repo with branch structure.
set -e

git init
git branch -M main

git checkout -b member/abdullah 2>/dev/null || git checkout member/abdullah
git checkout main
git checkout -b member/raza 2>/dev/null || git checkout member/raza
git checkout main
git checkout -b member/maarij 2>/dev/null || git checkout member/maarij
git checkout main

echo ""
echo "Branches created. Next steps:"
echo "  1. Create GitHub repo: pak-public-figures-classifier"
echo "  2. Add collaborators: asif370, omerrfarooqq, Aun-Dev146, ahsan608"
echo "  3. Protect 'main' branch (require PRs, 1 approving review)"
echo "  4. Run: bash scripts/distribute_commits.sh"
echo "  5. git remote add origin https://github.com/<USER>/pak-public-figures-classifier.git"
echo "  6. git push -u origin main"
echo "  7. git push origin member/abdullah member/raza member/maarij"
