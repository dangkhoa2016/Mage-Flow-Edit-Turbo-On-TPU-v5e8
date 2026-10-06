# Reproducibility

> 🌐 Language: **English** · [Tiếng Việt](REPRODUCIBILITY.vi.md)

Reproducibility claims are configuration-specific. The frozen production acceptance uses 512×512, q256, four steps and seeds 42, 43, 44, 45. Each accepted case produced four unique PNG hashes.

The 768/q256 repeat reproduced 4/4 PNG hashes bit-for-bit. The tested 1024/q128 path was also deterministic within q128. However, q128 and q256 are not cross-bit-exact at 512. Do not compare outputs across query-chunk settings as though they were the same qualified configuration.

Machine-readable authorities live in `acceptance/PRODUCTION_ACCEPTANCE.json` and `release-evidence/`.
