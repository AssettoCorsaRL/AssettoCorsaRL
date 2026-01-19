# Publishing to PyPI Guide

This guide explains how to publish the `assetto-corsa-rl` package to PyPI.

## Prerequisites

1. **Create PyPI Account**
   - Go to https://pypi.org/account/register/
   - Verify your email address

2. **Create Test PyPI Account** (optional, for testing)
   - Go to https://test.pypi.org/account/register/

3. **Install Build Tools**
   ```bash
   pip install --upgrade build twine
   ```

## Preparation

### 1. Update Version Number

Edit `pyproject.toml` and `setup.cfg`:

```toml
version = "0.1.0"  # Update to version
```

### 2. Update README.md

- Ensure all installation instructions are correct
- Update any URLs or repository links
- Add release notes if needed

### 3. Test Locally

```bash
# Install in development mode
pip install -e .

# Test the CLI
acrl --version
acrl --help

# Run tests
pytest tests/
```

## Building the Package

### 1. Clean Previous Builds

```bash
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist *.egg-info rmdir /s /q *.egg-info

rm -rf build dist *.egg-info
```

### 2. Build Distribution Files

```bash
python -m build
```

This creates:
- `dist/assetto_corsa_rl-0.1.0-py3-none-any.whl` (wheel)
- `dist/assetto-corsa-rl-0.1.0.tar.gz` (source distribution)

### 3. Verify the Build

```bash
# List contents
tar -tzf dist/assetto-corsa-rl-0.1.0.tar.gz

# Install locally to test
pip install dist/assetto_corsa_rl-0.1.0-py3-none-any.whl
```

## Publishing

### Option 1: Test PyPI (Recommended First)

```bash
python -m twine upload --repository testpypi dist/*

pip install --index-url https://test.pypi.org/simple/ assetto-corsa-rl
```

### Option 2: Production PyPI

```bash
python -m twine upload dist/*

```

### Option 3: Using API Token (Recommended)

1. **Create API Token**
   - Go to https://pypi.org/manage/account/token/
   - Create a new API token
   - Copy the token (starts with `pypi-`)

2. **Create `.pypirc` file**

Windows: `%USERPROFILE%\.pypirc`
Linux/Mac: `~/.pypirc`

```ini
[pypi]
username = __token__
password = pypi-your-api-token-here

[testpypi]
username = __token__
password = pypi-your-test-api-token-here
```

3. **Upload with Token**

```bash
python -m twine upload dist/*
```

## Post-Publication

### 1. Verify Installation

```bash
python -m venv test_env
source test_env/bin/activate 

pip install assetto-corsa-rl

# Test
acrl --version
acrl --help
```

### 2. Create Git Tag

```bash
git tag -a v0.1.0 -m "Release version 0.1.0"
git push origin v0.1.0
```

### 3. Create GitHub Release

- Go to repository releases page
- Create new release from tag
- Add release notes
- Attach distribution files (optional)

## Updating the Package

### 1. Update Version

Follow [Semantic Versioning](https://semver.org/):
- MAJOR version for incompatible API changes
- MINOR version for new functionality (backward compatible)
- PATCH version for bug fixes

### 2. Rebuild and Upload

```bash
# Clean, build, upload
rm -rf dist build
python -m build
python -m twine upload dist/*
```

## Troubleshooting

### Package Already Exists

Once uploaded, you cannot replace a version. You must:
1. Increment version number
2. Delete the old version from PyPI (if needed)
3. Upload new version

### Import Errors

Check that:
- `src/assetto_corsa_rl/__init__.py` exists
- Package structure matches `[tool.setuptools]` config
- All required files are included in MANIFEST.in

### Missing Dependencies

Users report missing packages:
1. Add to `dependencies` in pyproject.toml
2. Rebuild and upload new version

### CLI Command Not Found

Check that:
- Entry point is correct in pyproject.toml: `acrl = "assetto_corsa_rl.cli:main"`
- After installation, run: `pip show -f assetto-corsa-rl`

## Best Practices

1. **Always test on Test PyPI first**
2. **Use semantic versioning**
3. **Maintain a CHANGELOG**
4. **Tag releases in Git**
5. **Use API tokens instead of passwords**
6. **Document breaking changes clearly**
7. **Keep dependencies minimal**
