# Next.js Error Analysis

Analyze Next.js error logs and screenshots using Daft for pattern recognition and similarity search.

## Setup

```bash
pip install -r requirements.txt
python3 nextjs_error_analysis.py
```

## Usage

Place your error data in:

- `../nextjs_errors/logs/` - Text error logs
- `../nextjs_errors/screenshots/` - Error screenshots

## Features

- Multimodal analysis (text + images)
- Error classification and pattern detection
- Embedding-based similarity search
- Actionable recommendations

## Output

Processes errors, generates embeddings, and provides insights on:

- Authentication issues
- Hydration errors
- Runtime crashes
- Build failures
- Module resolution problems

## License

MIT
