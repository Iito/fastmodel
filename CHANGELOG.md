# CHANGELOG



## v0.2.0 (2025-06-27)

### Documentation

* docs: update documentation ([`81f44fb`](https://github.com/Iito/fastmodel/commit/81f44fb125538d9f156a4297f133f1dd0fefe154))

### Feature

* feat: pillow library issue in type checks whilst creating Request/Response model

- model_version was called before being created in case where the AI class had no version attribute.
- deleting pydantic model validator to create the response model ([`7f8e174`](https://github.com/Iito/fastmodel/commit/7f8e174edf51883283668363d426b5fdcbfe22b6))

* feat: Starting project ([`4749ca3`](https://github.com/Iito/fastmodel/commit/4749ca3c608a320a0eb0407185c4b7d109574cca))
