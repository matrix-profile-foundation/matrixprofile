# Deployment Instructions

## Source Code
The matrixprofile library consists a multi-stage TravisCI build. The steps to create a release are as follows:

1. Update the version.py
2. Update the docs/Releases.md
3. Push the commit with the release to master.

```
git commit version.py docs/Releases.md -m "release vX.X.X"
git push origin master
```

4. Now we create a git tag that triggers a TravisCI build.

```
git tag -a 'vX.X.X' -m 'release vX.X.X'
git push --tags
```

## API Documentation
The API documentation is hosted on Github pages. A bash script exists to "deploy" the code.

```
bash docs/deploy_docs.sh
```
