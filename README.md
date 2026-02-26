# zkbench

Benchmark library for zero-knowledge proof implementations.

## Installation

```bash
pip install zkbench
```

### Optional Dependencies

```bash
# Development tools
pip install zkbench[dev]
```

## Usage

```python
from zkbench import (
    BenchmarkReport,
    BenchmarkResult,
    Metadata,
    MetricValue,
    TestVectors,
    calculate_statistics,
    calculate_confidence_interval,
    compute_array_hash,
)

# Create benchmark metadata
metadata = Metadata.create(
    implementation="my-zk-impl",
    version="1.0.0",
)

# Record benchmark results
latency_values = [1.2, 1.3, 1.1, 1.4, 1.2]
mean, stdev = calculate_statistics(latency_values)
lower, upper = calculate_confidence_interval(mean, stdev, len(latency_values))

result = BenchmarkResult(
    latency=MetricValue(
        value=mean,
        unit="ms",
        lower_value=lower,
        upper_value=upper,
    ),
    iterations=len(latency_values),
    test_vectors=TestVectors(
        input_hash=compute_array_hash(input_data),
        output_hash=compute_array_hash(output_data),
        verified=True,
    ),
)

# Create and export report
report = BenchmarkReport(
    metadata=metadata,
    benchmarks={"poseidon_hash": result},
)

print(report.to_json())
```

## API Reference

### Data Classes

| Class             | Description                                             |
| ----------------- | ------------------------------------------------------- |
| `MetricValue`     | Benchmark metric with optional confidence bounds        |
| `TestVectors`     | Test vector verification (input/output hashes)          |
| `BenchmarkResult` | Single benchmark results (latency, memory, throughput)  |
| `Platform`        | Platform information (os, arch, cpu)                    |
| `Metadata`        | Benchmark metadata (implementation, version, timestamp) |
| `BenchmarkReport` | Complete benchmark report                               |

### Functions

| Function                                                    | Description                           |
| ----------------------------------------------------------- | ------------------------------------- |
| `calculate_statistics(values)`                              | Calculate mean and standard deviation |
| `calculate_confidence_interval(mean, stdev, n, confidence)` | Calculate confidence interval bounds  |
| `compute_array_hash(arr)`                                   | Compute SHA256 hash of array data     |
| `get_git_commit_sha()`                                      | Get current git commit SHA            |

## License

Apache License 2.0
