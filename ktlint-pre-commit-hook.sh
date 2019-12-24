#!/bin/sh
# https://github.com/shyiko/ktlint pre-commit hook
echo Reformatting code...
git diff --name-only --relative | grep '\.kt[s"]\?$' | xargs ./ktlint -F --relative . --disabled_rules=no-wildcard-imports,experimental:annotation
echo Code reformatted!
