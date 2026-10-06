# Limitations

> 🌐 Language: **English** · [Tiếng Việt](LIMITATIONS.vi.md)

- Production qualification is limited to 512×512 / q256 / four steps on the tested TPU v5e-8 topology.
- 768×768 / q256 is validated scaling, not the production baseline.
- 1024×1024 / q128 is experimental high resolution.
- q128 and q256 are not bit-identical at 512.
- Timing measurements are run-specific evidence, not service-level guarantees.
- CPU regression tests do not prove TPU numerical equivalence by themselves.
- This project is an independent runtime/conversion engineering effort; it does not retrain or relicense upstream model weights.
