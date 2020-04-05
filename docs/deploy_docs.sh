#!/bin/bash
set -e

SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
HTMLDIR="${SCRIPTPATH}/build/html"
MSG="deployed docs"

while getopts m: option
do
    case "${option}"
    in
        m) MSG=${OPTARG};;
    esac
done

make html
cd /tmp/

if [ -d "matrixprofile-docs-website" ]; then
    rm -rf matrixprofile-docs-website
fi


git clone git@github.com:matrix-profile-foundation/matrixprofile-docs-website.git
cd matrixprofile-docs-website
git checkout master
git rm -rf .
git clean -fxd
echo "matrixprofile.docs.matrixprofile.org" > CNAME
cp -R "${HTMLDIR}"/* .
git add .
git commit -am "${MSG}"
git push origin master
cd /
rm -rf /tmp/matrixprofile-docs-website
