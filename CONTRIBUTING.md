# Contributing to `blipss`

Contributions are always welcome, and they are greatly appreciated!

## Types of Contributions

### 1. Report Bugs

Report bugs [here on GitHub](https://github.com/UCBerkeleySETI/blipss/issues). If you are reporting a bug, please include:
- Your operating system name and version.
- Any details about your local setup that might be helpful in troubleshooting.
- Detailed steps to reproduce the bug.

### 2. Fix Bugs

Look through the GitHub issues for bugs.
Anything tagged with "bug" and "help wanted" is open to whoever wants to implement a fix for it.

### 3. Implement Features

Look through the GitHub issues for features.
Anything tagged with "enhancement" and "help wanted" is open to whoever wants to implement it.

### 4. Write Documentation

`blipss` could always use better documentation, whether as part of the official docs, in docstrings, or even on the web in blog posts, articles, and such.

### 5. Submit Feedback

The best way to provide constructive criticisms is to (file an issue on GitHub](https://github.com/UCBerkeleySETI/blipss/issues). If you are proposing a new feature:
- Explain in detail how it would work.
- Keep the scope as narrow as possible, to make it easier to implement.
- Remember that this is a volunteer-driven project, and that contributions are welcome.

## Get Started!

Ready to contribute? Here's how to set up `blipss` for local development.
Please note this documentation assumes you already have `uv` and `git` installed and ready to go.

1. Fork the `blipss` repo on GitHub.

2. Clone your fork locally:
```bash
cd <directory_in_which_repo_should_be_created>
git clone git@github.com:YOUR_NAME/blipss.git
```

3. Now we need to install the environment. Navigate into the root package directory, and then install the uv environment.

```bash
cd blipss
uv sync
```

4. Install pre-commit to run linters/formatters at commit time.
```bash
uv run pre-commit install
```

5. Create a branch for local development.
```bash
git checkout -b name-of-your-bugfix-or-feature
```
Now, you can make your changes locally.

6. Don't forget to add test cases for your added functionality to the `tests` directory.

7. When you're done making changes, check that your changes pass the formatting tests.
```bash
make check
```

Now, validate that all unit tests are passing:
```bash
make test
```

10. Commit your changes and push your branch to GitHub:
```bash
git add .
git commit -m "Your detailed description of your changes."
git push origin name-of-your-bugfix-or-feature
```

11. Submit a pull request through the GitHub website.

## Pull Request Guidelines

Before you submit a pull request, check that it meets these guidelines:

1. The pull request includes tests along with every feature update.

2. If the pull request adds CLI functionality, the README.md doc must be updated.
   Put your new functionality into a function with a docstring, and add the feature to the list in `README.md`.
