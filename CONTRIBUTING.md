# Contributing to Hybrid Intent Router

Thank you for your interest in contributing to Hybrid Intent Router! We welcome contributions from everyone. By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

## 🚀 How to Contribute

### Reporting Bugs
If you find a bug, please create a new issue with the following details:
- **Title**: A clear and concise description of the issue.
- **Description**: Steps to reproduce, expected behavior, and actual behavior.
- **Environment**: Python version, OS, and relevant package versions.
- **Logs**: Any error logs or stack traces.

### Suggesting Enhancements
We love new ideas! If you have a suggestion:
1. Check if it's already discussed in the [Issues](https://github.com/yourusername/hybrid-intent-router/issues).
2. Open a new issue with the tag `enhancement`.
3. Describe the feature and why it would be useful.

### Pull Requests (PR)
1. **Fork** the repository and clone it locally.
2. Create a new branch: `git checkout -b feature/amazing-feature`.
3. Make your changes and commit them: `git commit -m 'Add some amazing feature'`.
4. Push to the branch: `git push origin feature/amazing-feature`.
5. Open a **Pull Request**.

## 🛠 Development Setup

1. **Clone the repo**:
   ```bash
   git clone https://github.com/yourusername/hybrid-intent-router.git
   cd hybrid-intent-router
   ```

2. **Install dependencies** (using Poetry is recommended):
   ```bash
   poetry install
   ```
   Or using pip:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Tests**:
   ```bash
   pytest tests/
   ```

## 🎨 Code Style & Conventions

- **Python**: We use `black` for formatting and `isort` for import sorting.
  ```bash
  poetry run black .
  poetry run isort .
  ```
- **Commit Messages**: Please use clear and descriptive commit messages.
  - `feat: Add new router logic`
  - `fix: Resolve memory leak in embedding engine`
  - `docs: Update API documentation`

## 🏷 Good First Issues
If you're new to the project, check out issues tagged with `good first issue` or `help wanted`. These are great starting points!

Thank you for contributing!
