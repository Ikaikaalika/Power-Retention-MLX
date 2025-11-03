# Migration Guide: New Repository Structure

## ✅ Reorganization Complete!

The codebase has been reorganized into a clean, professional Python package structure.

## 📊 What Changed

### Directory Structure

**Before:**
```
Power-Retention-MLX/
├── power_retention.py         # Core (root level - messy)
├── *_example.py               # Examples scattered
├── test_power_retention.py    # Test (root level)
├── *.md                        # Docs scattered
└── llm_training/               # Isolated subdirectory
```

**After:**
```
power-retention-mlx/
├── src/power_retention/        # ✅ Core package (standard layout)
├── examples/                   # ✅ All examples organized
│   ├── basic/
│   ├── advanced/
│   └── llm/
├── tests/                      # ✅ All tests centralized
├── docs/                       # ✅ All docs organized
│   ├── guides/
│   └── llm/
└── llm/                        # ✅ Integrated as subpackage
```

## 🔄 Import Changes

### Core Module (Backward Compatible!)

✅ **No changes needed** - imports work exactly the same:

```python
from power_retention import PowerRetention
```

### LLM Training (Improved Paths)

**Before:**
```python
from llm_training.models import RetentionLLM
from llm_training.data import DataProcessor
```

**After:**
```python
from llm.models import RetentionLLM
from llm.data import DataProcessor
```

## 📁 File Moves

### Core Module
```
power_retention.py → src/power_retention/core.py
```

### Examples
```
simple_example.py     → examples/basic/simple_usage.py
model_example.py      → examples/basic/model_building.py
rl_integration.py     → examples/advanced/rl_integration.py
```

### Tests
```
test_power_retention.py → tests/test_power_retention.py
```

### Documentation
```
QUICK_START.md                    → docs/guides/QUICK_START.md
METAL_KERNELS.md                  → docs/guides/METAL_KERNELS.md
llm_training/README.md            → docs/llm/README.md
llm_training/GETTING_STARTED.md   → docs/llm/GETTING_STARTED.md
```

### LLM Package
```
llm_training/ → llm/
```

## 🚀 How to Use New Structure

### 1. Install/Reinstall Package

```bash
# Basic installation
pip install -e .

# With LLM training support
pip install -e ".[llm]"

# Development installation
pip install -e ".[dev]"
```

### 2. Run Examples

```bash
# Basic examples
python3 -m examples.basic.simple_usage
python3 -m examples.basic.model_building

# Advanced examples
python3 -m examples.advanced.rl_integration

# LLM examples
python3 -m examples.llm.quick_start
```

### 3. Run Tests

```bash
# All tests
pytest tests/

# With coverage
pytest tests/ --cov=src/power_retention
```

### 4. Access Documentation

```bash
# User guides
open docs/guides/QUICK_START.md
open docs/guides/METAL_KERNELS.md

# LLM training guides
open docs/llm/README.md
open docs/llm/GETTING_STARTED.md
```

## 🎯 Benefits

### 1. Standard Python Package
- ✅ Follows PEP 518/621 standards
- ✅ `src/` layout prevents import issues
- ✅ Clean pip installation
- ✅ Professional structure

### 2. Better Organization
- ✅ Examples grouped by complexity
- ✅ Documentation centralized
- ✅ Tests in dedicated directory
- ✅ LLM training integrated

### 3. Easier Navigation
- ✅ Find examples: `examples/`
- ✅ Find docs: `docs/`
- ✅ Find tests: `tests/`
- ✅ Find source: `src/power_retention/`

### 4. Development Friendly
- ✅ Run tests: `pytest tests/`
- ✅ Install dev: `pip install -e ".[dev]"`
- ✅ Type check: `mypy src/`
- ✅ Coverage: `pytest --cov`

### 5. Ready for Distribution
- ✅ PyPI-ready structure
- ✅ Proper package metadata
- ✅ Optional dependencies
- ✅ Professional appearance

## 📝 Updated pyproject.toml

New features:
- ✅ `src/` layout configuration
- ✅ Optional dependencies (`[llm]`, `[dev]`, `[all]`)
- ✅ Pytest configuration
- ✅ Enhanced metadata

## 🔧 Troubleshooting

### Import Error

If you get import errors:

```bash
# Reinstall in editable mode
pip install -e .

# Or add src/ to PYTHONPATH temporarily
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

### Examples Not Running

Use module syntax:
```bash
python3 -m examples.basic.simple_usage
```

Or run from examples directory:
```bash
cd examples/basic
python3 simple_usage.py
```

### Old Files Still Present

The old files are still there for reference. You can remove them:

```bash
# ⚠️ Only after verifying new structure works!
rm power_retention.py simple_example.py model_example.py rl_integration.py
rm test_power_retention.py
rm QUICK_START.md METAL_KERNELS.md
rm -rf llm_training/
```

## ✅ Verification Checklist

- [ ] Core module imports: `from power_retention import PowerRetention`
- [ ] Examples run from `examples/` directory
- [ ] Tests pass with `pytest tests/`
- [ ] Documentation accessible in `docs/`
- [ ] LLM package imports: `from llm.models import RetentionLLM`
- [ ] Package installs: `pip install -e .`

## 🎉 What's Next?

1. **Update GitHub README**: Replace with `README_NEW.md`
2. **Remove Old Files**: Clean up root directory (optional)
3. **Update CI/CD**: Configure for new structure (if applicable)
4. **Publish to PyPI**: Structure now ready for distribution!

## 📚 Resources

- **New README**: [README_NEW.md](README_NEW.md)
- **Quick Start**: [docs/guides/QUICK_START.md](docs/guides/QUICK_START.md)
- **LLM Training**: [docs/llm/README.md](docs/llm/README.md)
- **Project Structure**: [REORGANIZATION_PLAN.md](REORGANIZATION_PLAN.md)

---

**Questions or issues?** Check the documentation or open an issue on GitHub.
