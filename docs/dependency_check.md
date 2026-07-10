# Dependency Check

Checked on a clean Python 3.11 virtual environment.

## Base Package

```bash
python -m pip install -e .
python scripts/check_results.py
python scripts/audit_protocol.py --output-dir tmp/depcheck_protocol
python -m pip check
```

Result:

- table check: passed
- protocol audit: 14 rows, 0 failures
- pip check: no broken requirements

## Optional Experiment Dependencies

```bash
python -m pip install --dry-run -r requirements.txt
python -m pip install --dry-run -r requirements-brax.txt
```

Result:

- `requirements.txt` resolves PyTorch, Gymnasium, and their dependencies.
- `requirements-brax.txt` resolves Brax, JAX, MuJoCo/MJX, and their dependencies.

## Path Check

The repository was searched for machine-specific paths, API keys, and local
workspace names. The only expected hits are author email addresses and the
public GitHub URL.

