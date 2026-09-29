# Contributing to EmbedPlan

Thank you for your interest! EmbedPlan is meant to be used, not only reproduced, and we welcome
questions, bug reports, new domains, new encoders and results on your own data.

## Ways to help

- **Try it on your data** and tell us how it went in
  [Discussions](https://github.com/embedplan/EmbedPlan/discussions): the domain, the encoder, the
  numbers `evaluate()` printed, and anything that was harder than it should be.
- **Report a bug** with the [bug form](https://github.com/embedplan/EmbedPlan/issues/new/choose): what you
  ran, what you expected, what happened, and your versions.
- **Share a domain or benchmark**: a new planning domain, game, web or UI environment, or any
  source of text transitions. A loader in `embedplan/datasets.py` that returns `X, y, groups`
  lets everyone evaluate on it.
- **Add an encoder**: anything that maps a list of texts to an array works through
  `embedplan.encoders.get_encoder`. Named presets for new model families are welcome.
- **Improve the method**: search on top of the transition model, better generalization to unseen
  problems (the paper's open question), new objectives. Please open an issue first to discuss.

## Development setup

```bash
git clone https://github.com/embedplan/EmbedPlan.git
cd EmbedPlan
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q          # CPU only, synthetic data, about 20 seconds
ruff check .
```

## Pull requests

1. Open an issue (or comment on one) so we can agree on the approach.
2. Keep each pull request focused, and add a test under `tests/` for new behavior. Tests run on
   CPU with synthetic data and no downloads; `embedplan.datasets.load_toy_ferry` and
   `tools/make_toy_domain.py` give you data in seconds.
3. Keep the paper's numbers reproducible: code under `experiments/` and the evaluators in
   `embedplan/evaluation.py` and `embedplan/paper_protocol.py` must not change what they compute.
   Put new behavior behind a new function or flag.
4. Make sure `pytest -q` and `ruff check .` pass. CI runs both on Python 3.10 and 3.12.

## Code of conduct

This project follows our [Code of Conduct](CODE_OF_CONDUCT.md). By taking part you agree to it.
