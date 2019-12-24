#!/bin/sh
# https://github.com/shyiko/ktlint pre-commit hook
pwd
echo Reformatting code...
git diff --name-only --cached --relative | grep '\.kt[s"]\?$' | xargs ./ktlint -F --relative . --disabled_rules=no-wildcard-imports,experimental:annotation
if [ $? -ne 0 ]; then exit 1; fi
