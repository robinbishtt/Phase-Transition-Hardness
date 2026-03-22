# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability in this project, please report it by emailing:

**[ANONYMOUS_EMAIL]**

Please include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

We will respond within 48 hours and work to resolve the issue promptly.

## Security Considerations

This is a research codebase for computational experiments. Security considerations:

1. **No network operations**: The code does not make network requests
2. **No execution of user input**: User input is limited to configuration parameters
3. **File I/O**: Limited to specified directories (results/, data/)
4. **External solvers**: Optional integration with Kissat/CaDiCaL (user-installed)

## Best Practices

When using this code:
- Run in isolated environments (containers/VMs)
- Validate all input parameters
- Do not execute untrusted instance files
- Keep dependencies updated
