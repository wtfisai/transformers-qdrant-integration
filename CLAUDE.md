# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Essential Commands

### Development Setup
```bash
# Clone and setup Hugging Face transformers (if needed)
cd transformers
git remote add upstream https://github.com/huggingface/transformers.git

# Create virtual environment and install in development mode
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"  # Full development dependencies
# OR minimal setup:
pip install -e ".[quality]"  # Just linting/formatting tools
```

### Code Quality and Formatting
```bash
# Quick fix for modified files only (preferred for PRs)
cd transformers && make fixup

# Full style check and fix on all files
cd transformers && make style

# Check code quality without fixing
cd transformers && make quality

# Fix code copies and generated files
cd transformers && make fix-copies

# Check repository consistency
cd transformers && make repo-consistency
```

### Testing
```bash
# Run all tests
cd transformers && make test
# OR with pytest directly:
cd transformers && python -m pytest -n auto --dist=loadfile -s -v ./tests/

# Run specific test file or directory
cd transformers && pytest tests/models/bert/test_modeling_bert.py
cd transformers && pytest tests/models/bert/

# Run slow tests (comprehensive testing)
cd transformers && RUN_SLOW=1 pytest tests/

# Run a single test
cd transformers && pytest tests/models/bert/test_modeling_bert.py::BertModelTest::test_forward_signature

# Run example tests
cd transformers && make test-examples
```

### Building and Documentation
```bash
# Update dependency version table
cd transformers && make deps_table_update

# Build documentation locally
cd transformers && pip install -e ".[docs]"
cd transformers && pip install git+https://github.com/huggingface/doc-builder
cd transformers && doc-builder build transformers docs/source/en/ --build_dir ~/tmp/test-build

# Build release artifacts
cd transformers && make build-release
```

## Project Structure

This repository contains:
- `/transformers/` - The main Hugging Face Transformers library code
- `/qdrant_setup.py` - Script for setting up Qdrant vector database client
- `/secrets.txt` - Contains sensitive API keys (DO NOT COMMIT)
- `/bfg.jar` - Tool for removing sensitive data from git history

### Transformers Library Structure
```
transformers/src/transformers/
├── models/              # All model implementations (each model in its own directory)
│   └── bert/           # Example model directory
│       ├── __init__.py
│       ├── configuration_bert.py     # Model configuration
│       ├── modeling_bert.py          # PyTorch implementation
│       ├── modeling_tf_bert.py       # TensorFlow implementation
│       ├── modeling_flax_bert.py     # Flax/JAX implementation
│       └── tokenization_bert.py      # Tokenizer implementation
├── pipelines/          # High-level API for common tasks
├── generation/         # Text generation utilities
├── data/              # Data processing utilities
├── integrations/      # External library integrations
├── quantizers/        # Model quantization implementations
└── utils/             # Shared utilities
```

## Key Design Patterns

1. **Multi-Framework Support**: Each model can have PyTorch, TensorFlow, and Flax implementations. The library uses lazy loading to only import the framework you need.

2. **Configuration-Model Separation**: Every model has a configuration class that defines its architecture, separate from the model implementation. This allows saving/loading model configurations independently.

3. **Tokenizer Architecture**: Tokenizers are separate from models and can be mixed and matched. Fast tokenizers (Rust-based) are preferred when available.

4. **Pipeline Abstraction**: The `pipeline()` function provides a high-level API that automatically handles model loading, preprocessing, and postprocessing for common tasks.

5. **Model Hub Integration**: Deep integration with Hugging Face Hub for model sharing and downloading. Models are automatically cached locally.

## Code Organization Principles

1. **Self-Contained Models**: Each model's code is intentionally not heavily abstracted. This allows researchers to modify individual models without affecting others.

2. **Copy Mechanism**: Common code patterns are maintained through a copy mechanism (marked with `# Copied from`) rather than inheritance. This is checked by `make fix-copies`.

3. **Lazy Imports**: The library uses `_LazyModule` to defer imports, improving import time and allowing users to use the library without all dependencies.

4. **Modular Transformers**: New models can use the modular system where components are defined in separate files and composed together.

## Testing Philosophy

- **Fast vs Slow Tests**: Tests are separated into fast (run by default) and slow (comprehensive, run with `RUN_SLOW=1`)
- **Model Tests**: Each model has standardized tests that are automatically generated from common test mixins
- **Integration Tests**: Separate tests for pipelines, trainer, and other high-level components
- **Example Tests**: All example scripts are tested to ensure they remain functional

## Development Workflow

1. **Feature Branch**: Create a descriptive branch name
2. **Make Changes**: Implement your feature/fix
3. **Run Tests**: Test your specific changes
4. **Format Code**: Run `make fixup` for smart formatting of modified files
5. **Check Quality**: Ensure `make quality` passes
6. **Repository Consistency**: Run `make repo-consistency`
7. **Commit**: Make atomic commits with clear messages

## Important Configuration Files

- `transformers/setup.py`: Main package configuration, dependencies, and entry points
- `transformers/pyproject.toml`: Tool configurations (ruff, pytest settings)
- `transformers/Makefile`: Development workflow automation
- `transformers/src/transformers/dependency_versions_table.py`: Auto-generated dependency versions

## Framework-Specific Imports

When working with framework-specific code:
- PyTorch: Import from `transformers.models.{model_name}.modeling_{model_name}`
- TensorFlow: Import from `transformers.models.{model_name}.modeling_tf_{model_name}`
- Flax: Import from `transformers.models.{model_name}.modeling_flax_{model_name}`

## Adding New Models

New models should follow the existing patterns:
1. Create a new directory under `transformers/src/transformers/models/`
2. Implement configuration, tokenizer, and model files
3. Add model to relevant `__init__.py` files and auto mappings
4. Write comprehensive tests
5. Add documentation
6. Update model cards and examples

The library provides templates and utilities to help with this process.

## Working with Qdrant Integration

This codebase includes a Qdrant vector database setup script (`qdrant_setup.py`). The script contains API credentials for connecting to a Qdrant cloud instance. When working with this integration:
- The API key and URL are hardcoded in the script
- Use the script to test connectivity and list available collections
- Do not commit any changes that expose additional credentials

## Security Notes

- The repository contains `secrets.txt` which should NEVER be committed
- API keys in `qdrant_setup.py` are already exposed in the repository
- Use `bfg.jar` if you need to clean sensitive data from git history
- Always run `git status` before committing to ensure no sensitive files are staged