# Security Policy

This repository is a research benchmark for generative models of financial time
series. It ships model code, evaluation metrics, and reproduction scripts. It
does not run a hosted service and does not process third party user data.

## Supported versions

Security fixes are applied to the `master` branch only. There is no long term
support branch; please track `master` for the latest state.

## Reporting a vulnerability

If you discover a security issue (for example a supply chain risk in a pinned
dependency, an unsafe deserialization path in a data loader, or leaked
credentials in the git history), please report it privately rather than opening
a public issue.

- Email: tbasseras@murex.com
- Subject line: `SECURITY: time-series-generation-benchmark`

Please include a description of the issue, the file or dependency involved, and
a minimal way to reproduce it. We aim to acknowledge a report within five
business days and to agree on a disclosure timeline with the reporter.

## Scope

In scope:

- Unsafe code execution triggered by loading a shipped weight, dataset, or
  config file.
- Hardcoded secrets or credentials committed anywhere in the tree or its
  history.
- Dependency vulnerabilities that affect the pinned environments described in
  the method and metrics READMEs.

Out of scope:

- The statistical quality of any generative method (that is a research
  question, not a security issue). Use a normal issue for that.
- Vulnerabilities in third party upstream repositories vendored under
  `methods/*/code/reference/`. Please report those to their upstream projects.

## Acknowledgements

This benchmark is sponsored by Murex. Coordinated disclosures are credited in
the release notes unless the reporter prefers to stay anonymous.
