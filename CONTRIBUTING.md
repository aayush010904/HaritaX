# Contributing to HaritaX

Thank you for your interest in contributing to HaritaX! We welcome contributions from the community to make this project even better.

## 🤝 How Can You Contribute?

- 🐛 Reporting bugs
- 💡 Suggesting features or improvements
- 📚 Improving documentation
- 🔬 Adding new models or datasets
- ⚡ Refactoring code or improving performance
- 🧪 Writing tests

## 🚀 Getting Started

### 1. Fork the Repository
Click the 'Fork' button on the top right of this page.

### 2. Clone Your Fork
```
git clone https://github.com/your-username/haritax.git
cd haritax
```

### 3. Set Up Development Environment
```
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 4. Create a New Branch
```
git checkout -b feature/your-feature-name
```

### 5. Make Your Changes
Please ensure your code follows the existing style and includes appropriate tests.

### 6. Run Tests
```
python -m pytest tests/
```

### 7. Commit Your Changes
```
git add .
git commit -m "feat: add your descriptive commit message"
```

### 8. Push to Your Fork
```
git push origin feature/your-feature-name
```

### 9. Open a Pull Request
Go to the original repository and open a Pull Request from your branch.

## 📝 Development Guidelines

### Code Style
- Follow PEP8 for Python code
- Use type hints where appropriate
- Write clear, concise docstrings
- Keep functions small and focused

### Commit Messages
We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation changes
- `test:` for adding tests
- `refactor:` for code refactoring

### Testing
- Write unit tests for new functionality
- Ensure all tests pass before submitting PR
- Aim for good test coverage

### Documentation
- Update README.md if needed
- Add docstrings to new functions/classes
- Update relevant documentation files

## 🐛 Reporting Issues

When reporting bugs, please include:

1. **Description**: Clear description of the issue
2. **Steps to Reproduce**: Detailed steps to reproduce the bug
3. **Expected Behavior**: What you expected to happen
4. **Actual Behavior**: What actually happened
5. **Environment**: Python version, OS, package versions
6. **Screenshots**: If applicable

## 💡 Feature Requests

For feature requests, please:

1. Check if the feature already exists
2. Describe the feature in detail
3. Explain the use case and benefits
4. Provide examples if possible

## 📋 Pull Request Process

1. **Update Documentation**: Ensure documentation is updated
2. **Add Tests**: Include tests for new functionality
3. **Check CI**: Ensure all CI checks pass
4. **Review Ready**: Request review from maintainers

### PR Checklist
- [ ] Code follows style guidelines
- [ ] Tests added/updated and passing
- [ ] Documentation updated
- [ ] Commit messages follow convention
- [ ] No merge conflicts

## 🏷️ Labels

We use the following labels to categorize issues and PRs:

- `bug`: Something isn't working
- `enhancement`: New feature or request
- `documentation`: Improvements to documentation
- `good first issue`: Good for newcomers
- `help wanted`: Extra attention is needed

## 🤔 Questions?

If you have questions about contributing, feel free to:

- Open an issue with the `question` label
- Contact the maintainers
- Join our discussions

## 🙏 Recognition

All contributors will be recognized in our [Contributors](CONTRIBUTORS.md) file.

Thank you for helping improve HaritaX! 🌱

